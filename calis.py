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
from unet_model import UNet

class SupervisedUNetTrainer:
    def __init__(self, signal_to_image_checkpoint, image_to_signal_checkpoint, device='cuda'):
        self.device = device
        # UNet para señal -> imagen (A) - Esta se entrena
        self.unet_A = UNet(in_channels=1, out_channels=1).to(device)
        # UNet para imagen -> señal (B) - Esta se usa como supervisor
        self.unet_B = UNet(in_channels=1, out_channels=1).to(device)
        
        # Cargar pesos pre-entrenados
        print("Loading pre-trained weights...")
        # Cargar UNet A (señal -> imagen)
        checkpoint_A = torch.load(signal_to_image_checkpoint, map_location=device)
        if 'state_dict' in checkpoint_A:
            self.unet_A.load_state_dict(checkpoint_A['state_dict'])
        else:
            self.unet_A.load_state_dict(checkpoint_A)
        print(f"Loaded signal->image model from {signal_to_image_checkpoint}")
        
        # Cargar UNet B (imagen -> señal) y congelar sus pesos
        checkpoint_B = torch.load(image_to_signal_checkpoint, map_location=device)
        if 'state_dict' in checkpoint_B:
            self.unet_B.load_state_dict(checkpoint_B['state_dict'])
        else:
            self.unet_B.load_state_dict(checkpoint_B)
        print(f"Loaded image->signal model from {image_to_signal_checkpoint}")
        
        # Congelar los pesos de UNet B
        for param in self.unet_B.parameters():
            param.requires_grad = False
        self.unet_B.eval()
        
        # Optimizador solo para UNet A
        self.opt_A = optim.Adam(self.unet_A.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt_A, patience=5, factor=0.5)
        
        # Criterio
        self.criterion = nn.MSELoss()
        
        # Pesos de las pérdidas
        self.lambda_direct = 1.0    # Peso para pérdida de reconstrucción de imagen
        self.lambda_physical = 0.5  # Peso para pérdida de consistencia física
        
    def train(self, train_loader, val_loader, num_epochs=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'training_results_supervised_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/photoacoustic_supervised_{timestamp}')
        
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'val_loss': [],
            'loss_direct': [], 'loss_physical': []
        }
        
        for epoch in range(num_epochs):
            # Modo entrenamiento para UNet A
            self.unet_A.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as t:
                for signal, image in t:
                    signal, image = signal.to(self.device), signal.to(self.device)
                    
                    # Zero gradients
                    self.opt_A.zero_grad()
                    
                    # Forward pass
                    reconstructed_image = self.unet_A(signal)
                    
                    # Pérdida directa: qué tan bien reconstruye la imagen
                    loss_direct = self.criterion(reconstructed_image, image)
                    
                    # Pérdida física: la señal que genera la imagen reconstruida 
                    # debe parecerse a la señal original
                    predicted_signal = self.unet_B(reconstructed_image)
                    loss_physical = self.criterion(predicted_signal, signal)
                    
                    # Pérdida total
                    total_loss = (self.lambda_direct * loss_direct + 
                                self.lambda_physical * loss_physical)
                    
                    # Backward y optimización
                    total_loss.backward()
                    self.opt_A.step()
                    
                    train_loss += total_loss.item()
                    
                    # Actualizar barra de progreso
                    t.set_postfix({
                        'loss': total_loss.item(),
                        'loss_direct': loss_direct.item(),
                        'loss_physical': loss_physical.item()
                    })
            
            train_loss /= len(train_loader)
            
            # Validación
            val_loss, val_metrics = self.validate(val_loader)
            
            # Actualizar scheduler
            self.scheduler.step(val_loss)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'unet_A_state_dict': self.unet_A.state_dict(),
                    'opt_A_state_dict': self.opt_A.state_dict(),
                }, f'{save_dir}/best_model.pth')
            
            # Logging
            print(f'Epoch {epoch+1}:')
            print(f'Train Loss = {train_loss:.6f}')
            print(f'Val Loss = {val_loss:.6f}')
            print(f'Val Metrics: {val_metrics}')
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Loss/direct', val_metrics['loss_direct'], epoch)
            writer.add_scalar('Loss/physical', val_metrics['loss_physical'], epoch)
            
            # Guardar histórico
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['loss_direct'].append(val_metrics['loss_direct'])
            history['loss_physical'].append(val_metrics['loss_physical'])
            
            # Visualización periódica
            if (epoch + 1) % 10 == 0:
                self.visualize_results(val_loader, epoch, save_dir)
        
        self.save_training_history(history, save_dir)
        return history
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.unet_A.eval()
        total_loss = 0
        metrics = {'loss_direct': 0, 'loss_physical': 0}
        
        for signal, image in val_loader:
            signal, image = signal.to(self.device), image.to(self.device)
            
            # Forward passes
            reconstructed_image = self.unet_A(signal)
            predicted_signal = self.unet_B(reconstructed_image)
            
            # Calcular pérdidas
            loss_direct = self.criterion(reconstructed_image, image)
            loss_physical = self.criterion(predicted_signal, signal)
            
            total_loss += (self.lambda_direct * loss_direct + 
                         self.lambda_physical * loss_physical).item()
            
            metrics['loss_direct'] += loss_direct.item()
            metrics['loss_physical'] += loss_physical.item()
        
        # Promediar métricas
        total_loss /= len(val_loader)
        for key in metrics:
            metrics[key] /= len(val_loader)
        
        return total_loss, metrics
    
    def visualize_results(self, val_loader, epoch, save_dir):
        self.unet_A.eval()
        
        with torch.no_grad():
            signal, image = next(iter(val_loader))
            signal, image = signal.to(self.device), image.to(self.device)
            
            # Forward passes
            reconstructed_image = self.unet_A(signal)
            predicted_signal = self.unet_B(reconstructed_image)
            
            plt.figure(figsize=(15, 5))
            
            # Señal original
            plt.subplot(141)
            plt.imshow(signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Original Signal')
            plt.colorbar()
            
            # Imagen objetivo
            plt.subplot(142)
            plt.imshow(image.cpu().numpy()[0, 0], cmap='gray')
            plt.title('Target Image')
            plt.colorbar()
            
            # Imagen reconstruida
            plt.subplot(143)
            plt.imshow(reconstructed_image.cpu().numpy()[0, 0], cmap='gray')
            plt.title('Reconstructed Image')
            plt.colorbar()
            
            # Señal predicha de la imagen reconstruida
            plt.subplot(144)
            plt.imshow(predicted_signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Physical Validation\n(Predicted Signal)')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/epoch_{epoch+1}_results.png')
            plt.close()
    
    def save_training_history(self, history, save_dir):
        # Guardar histórico en numpy
        for key, value in history.items():
            np.save(f'{save_dir}/{key}.npy', np.array(value))
        
        # Graficar pérdidas
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(history['loss_direct'], label='Direct Loss')
        plt.plot(history['loss_physical'], label='Physical Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Component Losses')
        plt.legend()
        
        plt.savefig(f'{save_dir}/training_history.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train UNet A with UNet B as supervisor.")
    parser.add_argument('--num_epochs', type=int, default=100,
                       help="Number of epochs to train the model. (default: 100)")
    parser.add_argument('--signal_to_image_checkpoint', type=str, required=True,
                       help="Path to the pre-trained signal->image UNet checkpoint")
    parser.add_argument('--image_to_signal_checkpoint', type=str, required=True,
                       help="Path to the pre-trained image->signal UNet checkpoint")
    parser.add_argument('--batch_size', type=int, default=4,
                       help="Batch size for training (default: 4)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar datos
    train_loader, val_loader, test_loader = load_and_preprocess_data(
        "simulated_data", 
        batch_size=args.batch_size
    )
    
    # Inicializar y entrenar
    trainer = SupervisedUNetTrainer(
        signal_to_image_checkpoint=args.signal_to_image_checkpoint,
        image_to_signal_checkpoint=args.image_to_signal_checkpoint,
        device=device
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )

if __name__ == "__main__":
    main()