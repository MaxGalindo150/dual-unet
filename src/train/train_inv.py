import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from preprocess.preprocess_simulated_data import load_and_preprocess_data

from models.unet_model import UNet

def get_model(model_name, device='cuda'):
    if model_name == 'unet':
        return UNet(in_channels=1, out_channels=1).to(device)
    else:
        raise ValueError("Currently only supporting 'unet' for inverse problem")

def train_inverse_model(model_name, train_loader, val_loader, num_epochs=100, device='cuda'):
    # Inicializar modelo
    model = get_model(model_name, device)
    
    # Configurar directorios y logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'training_results_inverse_{model_name}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(f'runs/photoacoustic_inverse_{timestamp}')
    
    # Criterio y optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as t:
            for batch_idx, (signal, image) in enumerate(t):
                # Invertimos el orden: ahora la imagen es la entrada y la señal es el objetivo
                image, signal = image.to(device), signal.to(device)
                
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, signal)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                t.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for signal, image in val_loader:
                image, signal = image.to(device), signal.to(device)
                output = model(image)
                loss = criterion(output, signal)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
        
        # Logging
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        
        # Visualización periódica
        if (epoch + 1) % 10 == 0:
            visualize_inverse_results(model, val_loader, device, epoch, save_dir)
        
    save_training_history(train_losses, val_losses, save_dir)
    
    return model, train_losses, val_losses

def visualize_inverse_results(model, val_loader, device, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        signal, image = next(iter(val_loader))
        image = image.to(device)
        output = model(image)
        
        # Crear subplot con colormap específico para señales
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(image.cpu().numpy()[0, 0], cmap='gray')
        plt.title('Input Image')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(signal.numpy()[0, 0], cmap='viridis')
        plt.title('Ground Truth Signal')
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow(output.cpu().numpy()[0, 0], cmap='viridis')
        plt.title('Predicted Signal')
        plt.colorbar()
        
        plt.savefig(f'{save_dir}/epoch_{epoch+1}_samples.png')
        plt.close()

def save_training_history(train_losses, val_losses, save_dir):
    np.save(f'{save_dir}/train_losses.npy', np.array(train_losses))
    np.save(f'{save_dir}/val_losses.npy', np.array(val_losses))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss_curves.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train inverse model for image to signal mapping.")
    parser.add_argument('--model_name', type=str, default='unet',
                       help="Name of the model to train (currently only 'unet' supported).")
    parser.add_argument('--num_epochs', type=int, default=100,
                       help="Number of epochs to train the model. (default: 100)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar los datos (asumiendo que tienes la misma función load_and_preprocess_data)
    train_loader, val_loader, test_loader = load_and_preprocess_data("simulated_data")
    
    model, train_losses, val_losses = train_inverse_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=device
    )

if __name__ == "__main__":
    main()