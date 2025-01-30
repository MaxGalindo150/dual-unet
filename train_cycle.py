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

class CycleUNetTrainer:
    def __init__(self, signal_to_image_checkpoint, image_to_signal_checkpoint, device='cuda'):
        self.device = device
        # UNet para señal -> imagen (A)
        self.unet_A = UNet(in_channels=1, out_channels=1).to(device)
        # UNet para imagen -> señal (B)
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
        
        # Cargar UNet B (imagen -> señal)
        checkpoint_B = torch.load(image_to_signal_checkpoint, map_location=device)
        if 'state_dict' in checkpoint_B:
            self.unet_B.load_state_dict(checkpoint_B['state_dict'])
        else:
            self.unet_B.load_state_dict(checkpoint_B)
        print(f"Loaded image->signal model from {image_to_signal_checkpoint}")
        
        # Optimizadores
        self.opt_A = optim.Adam(self.unet_A.parameters(), lr=0.001)
        self.opt_B = optim.Adam(self.unet_B.parameters(), lr=0.001)
        
        # Schedulers
        self.sched_A = optim.lr_scheduler.ReduceLROnPlateau(self.opt_A, patience=5, factor=0.5)
        self.sched_B = optim.lr_scheduler.ReduceLROnPlateau(self.opt_B, patience=5, factor=0.5)
        
        # Criterio
        self.criterion = nn.MSELoss()
        
        # Pesos de las pérdidas
        self.lambda_A = 1.0  # Peso para pérdida directa A (señal->imagen)
        self.lambda_B = 0.7  # Peso para pérdida directa B (imagen->señal)
        self.lambda_cycle = 0.3  # Peso para pérdida de ciclo
        
    def train(self, train_loader, val_loader, num_epochs=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'training_results_cycle_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/photoacoustic_cycle_{timestamp}')
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 
                  'loss_A': [], 'loss_B': [], 'loss_cycle': []}
        
        for epoch in range(num_epochs):
            # Modo entrenamiento
            self.unet_A.train()
            self.unet_B.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as t:
                for signal, image in t:
                    signal, image = signal.to(self.device), image.to(self.device)
                    
                    # Zero gradients
                    self.opt_A.zero_grad()
                    self.opt_B.zero_grad()
                    
                    # Forward passes
                    # Ciclo señal -> imagen -> señal
                    fake_image = self.unet_A(signal)
                    recovered_signal = self.unet_B(fake_image)
                    
                    # Ciclo imagen -> señal -> imagen
                    fake_signal = self.unet_B(image)
                    recovered_image = self.unet_A(fake_signal)
                    
                    # Calcular pérdidas
                    loss_A = self.criterion(fake_image, image)
                    loss_B = self.criterion(fake_signal, signal)
                    loss_cycle_signal = self.criterion(recovered_signal, signal)
                    loss_cycle_image = self.criterion(recovered_image, image)
                    loss_cycle = (loss_cycle_signal + loss_cycle_image) / 2
                    
                    # Pérdida total
                    total_loss = (self.lambda_A * loss_A + 
                                self.lambda_B * loss_B + 
                                self.lambda_cycle * loss_cycle)
                    
                    # Backward y optimización
                    total_loss.backward()
                    self.opt_A.step()
                    self.opt_B.step()
                    
                    train_loss += total_loss.item()
                    
                    # Actualizar barra de progreso
                    t.set_postfix({
                        'loss': total_loss.item(),
                        'loss_A': loss_A.item(),
                        'loss_B': loss_B.item(),
                        'loss_cycle': loss_cycle.item()
                    })
            
            train_loss /= len(train_loader)
            
            # Validación
            val_loss, val_metrics = self.validate(val_loader)
            
            # Actualizar schedulers
            self.sched_A.step(val_loss)
            self.sched_B.step(val_loss)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'unet_A_state_dict': self.unet_A.state_dict(),
                    'unet_B_state_dict': self.unet_B.state_dict(),
                    'opt_A_state_dict': self.opt_A.state_dict(),
                    'opt_B_state_dict': self.opt_B.state_dict(),
                }, f'{save_dir}/best_model.pth')
            
            # Logging
            print(f'Epoch {epoch+1}:')
            print(f'Train Loss = {train_loss:.6f}')
            print(f'Val Loss = {val_loss:.6f}')
            print(f'Val Metrics: {val_metrics}')
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Loss/loss_A', val_metrics['loss_A'], epoch)
            writer.add_scalar('Loss/loss_B', val_metrics['loss_B'], epoch)
            writer.add_scalar('Loss/loss_cycle', val_metrics['loss_cycle'], epoch)
            
            # Guardar histórico
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['loss_A'].append(val_metrics['loss_A'])
            history['loss_B'].append(val_metrics['loss_B'])
            history['loss_cycle'].append(val_metrics['loss_cycle'])
            
            # Visualización periódica
            if (epoch + 1) % 10 == 0:
                self.visualize_results(val_loader, epoch, save_dir)
        
        self.save_training_history(history, save_dir)
        return history
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.unet_A.eval()
        self.unet_B.eval()
        total_loss = 0
        metrics = {'loss_A': 0, 'loss_B': 0, 'loss_cycle': 0}
        
        for signal, image in val_loader:
            signal, image = signal.to(self.device), image.to(self.device)
            
            # Forward passes
            fake_image = self.unet_A(signal)
            recovered_signal = self.unet_B(fake_image)
            fake_signal = self.unet_B(image)
            recovered_image = self.unet_A(fake_signal)
            
            # Calcular pérdidas
            loss_A = self.criterion(fake_image, image)
            loss_B = self.criterion(fake_signal, signal)
            loss_cycle = (self.criterion(recovered_signal, signal) + 
                         self.criterion(recovered_image, image)) / 2
            
            total_loss += (self.lambda_A * loss_A + 
                         self.lambda_B * loss_B + 
                         self.lambda_cycle * loss_cycle).item()
            
            metrics['loss_A'] += loss_A.item()
            metrics['loss_B'] += loss_B.item()
            metrics['loss_cycle'] += loss_cycle.item()
        
        # Promediar métricas
        total_loss /= len(val_loader)
        for key in metrics:
            metrics[key] /= len(val_loader)
        
        return total_loss, metrics
    
    def visualize_results(self, val_loader, epoch, save_dir):
        self.unet_A.eval()
        self.unet_B.eval()
        
        with torch.no_grad():
            signal, image = next(iter(val_loader))
            signal, image = signal.to(self.device), image.to(self.device)
            
            # Forward passes
            fake_image = self.unet_A(signal)
            recovered_signal = self.unet_B(fake_image)
            fake_signal = self.unet_B(image)
            recovered_image = self.unet_A(fake_signal)
            
            # Visualizar ciclo señal -> imagen -> señal
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Original Signal')
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(fake_image.cpu().numpy()[0, 0], cmap='gray')
            plt.title('Generated Image')
            plt.colorbar()
            
            plt.subplot(133)
            plt.imshow(recovered_signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Recovered Signal')
            plt.colorbar()
            
            plt.savefig(f'{save_dir}/epoch_{epoch+1}_cycle_signal.png')
            plt.close()
            
            # Visualizar ciclo imagen -> señal -> imagen
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(image.cpu().numpy()[0, 0], cmap='gray')
            plt.title('Original Image')
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(fake_signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Generated Signal')
            plt.colorbar()
            
            plt.subplot(133)
            plt.imshow(recovered_image.cpu().numpy()[0, 0], cmap='gray')
            plt.title('Recovered Image')
            plt.colorbar()
            
            plt.savefig(f'{save_dir}/epoch_{epoch+1}_cycle_image.png')
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
        plt.plot(history['loss_A'], label='Loss A')
        plt.plot(history['loss_B'], label='Loss B')
        plt.plot(history['loss_cycle'], label='Loss Cycle')
        plt.xlabel('Epoch')
        plt.ylabel('Component Losses')
        plt.legend()
        
        plt.savefig(f'{save_dir}/training_history.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train coupled UNets with cycle consistency.")
    parser.add_argument('--num_epochs', type=int, default=100,
                       help="Number of epochs to train the models. (default: 100)")
    parser.add_argument('--signal_to_image_checkpoint', type=str, required=True,
                       help="Path to the pre-trained signal->image UNet checkpoint")
    parser.add_argument('--image_to_signal_checkpoint', type=str, required=True,
                       help="Path to the pre-trained image->signal UNet checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar datos
    train_loader, val_loader, test_loader = load_and_preprocess_data("simulated_data")
    
    # Inicializar y entrenar
    trainer = CycleUNetTrainer(
        signal_to_image_checkpoint=args.signal_to_image_checkpoint,
        image_to_signal_checkpoint=args.image_to_signal_checkpoint,
        device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )
def main():
    parser = argparse.ArgumentParser(description="Train coupled UNets with cycle consistency.")
    parser.add_argument('--num_epochs', type=int, default=100,
                       help="Number of epochs to train the models. (default: 100)")
    parser.add_argument('--signal_to_image_checkpoint', type=str, required=True,
                       help="Path to the pre-trained signal->image UNet checkpoint")
    parser.add_argument('--image_to_signal_checkpoint', type=str, required=True,
                       help="Path to the pre-trained image->signal UNet checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar datos
    train_loader, val_loader, test_loader = load_and_preprocess_data("simulated_data")
    
    # Inicializar y entrenar
    trainer = CycleUNetTrainer(
        signal_to_image_checkpoint=args.signal_to_image_checkpoint,
        image_to_signal_checkpoint=args.image_to_signal_checkpoint,
        device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )

if __name__ == "__main__":
    main()