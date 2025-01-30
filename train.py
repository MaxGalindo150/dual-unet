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

from unet_model import UNet
from attention_unet_model import AttentionUNet
from physics_informed import SA_UNet
from preprocess.preprocess_simulated_data import load_and_preprocess_data

def get_model(model_name, device='cuda'):
    if model_name == 'unet':
        return UNet(in_channels=1, out_channels=1).to(device)
    elif model_name == 'attention_unet':
        return AttentionUNet(in_channels=1, out_channels=1).to(device)
    elif model_name == 'sa_unet':
        return SA_UNet().to(device)
    else:
        raise ValueError("Model name must be 'unet', 'attention_unet', or 'sa_unet'.")

def train_model(model_name, train_loader, val_loader, num_epochs=100, device='cuda'):
    # Inicializar modelo
    model = get_model(model_name, device)
    
    # Configurar directorios y logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'training_results_{model_name}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(f'runs/photoacoustic_reconstruction_{timestamp}')
    
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
            for batch_idx, (data, target) in enumerate(t):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()

                
                # Para modelos normales
                output = model(data)
                loss = criterion(output, target)
                
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
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                    
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
            visualize_results(model, val_loader, device, epoch, save_dir, model_name)
    
    # Guardar históricos y gráficas
    save_training_history(train_losses, val_losses, save_dir)
    
    return model, train_losses, val_losses

def visualize_results(model, val_loader, device, epoch, save_dir, model_name):
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(val_loader))
        sample_data = sample_data.to(device)
        

        sample_output = model(sample_data)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(sample_data.cpu().numpy()[0, 0])
        plt.title('Input')
        plt.subplot(132)
        plt.imshow(sample_target.numpy()[0, 0])
        plt.title('Ground Truth')
        plt.subplot(133)
        plt.imshow(sample_output.cpu().numpy()[0, 0])
        plt.title('Prediction')
        
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
    parser = argparse.ArgumentParser(description="Train a model for photoacoustic image reconstruction.")
    parser.add_argument('--model_name', type=str, required=True, 
                       choices=['unet', 'attention_unet', 'sa_unet'],
                       help="Name of the model to train.")
    parser.add_argument('--num_epochs', type=int, default=100,
                       help="Number of epochs to train the model. (default: 100)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = load_and_preprocess_data("simulated_data")
    
    model, train_losses, val_losses = train_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=device
    )

if __name__ == "__main__":
    main()