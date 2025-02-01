import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from src.preprocess import load_and_preprocess_data
from src.models.unet_model import UNet

def set_seed(seed):
    """
    Set all seeds for reproducibility
    
    Args:
        seed (int): Seed value to use
    """
    import torch
    import numpy as np
    import random
    import os
    
    # Python
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Extra CUDA settings for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Establecer semilla para operaciones de CPU en algunos casos
    os.environ['PYTHONHASHSEED'] = str(seed)

def calculate_structural_loss(pred, target):
    """Calcula pérdida estructural en el espacio de tensores"""
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
    target_norm = (target - target.min()) / (target.max() - target.min())
    return F.mse_loss(pred_norm, target_norm) + (1 - F.cosine_similarity(pred_norm.flatten(), target_norm.flatten(), dim=0))

def calculate_similarity_penalty(output, input_signal):
    """Penaliza si la salida se parece demasiado a la entrada (en tensores)"""
    output_norm = (output - output.mean()) / (output.std() + 1e-8)
    input_norm = (input_signal - input_signal.mean()) / (input_signal.std() + 1e-8)
    similarity = F.cosine_similarity(output_norm.flatten(), input_norm.flatten(), dim=0)
    return similarity.pow(2)



class SupervisedUNetTrainer:
    def __init__(self,config):
        self.device = config.get('device', 'cuda')
        self.unet_A = UNet(in_channels=1, out_channels=1).to(self.device)
        self.unet_B = UNet(in_channels=1, out_channels=1).to(self.device)
        
        # Cargar pesos pre-entrenados
        print("Loading pre-trained weights...")
        # Cargar UNet A (señal -> imagen)
        self._load_checkpoints(config['signal_to_image_checkpoint'], config['image_to_signal_checkpoint'])
        
        
        self.loss_weights = {
            'direct': config.get('lambda_direct', 1.0),
            'physical': config.get('lambda_physical', 0.1),
            'struct': config.get('lambda_struct', 2.0),
            'similarity': config.get('lambda_similarity', 0.3)
        }
        
        print("Experiment configuration:")
        print(f"Loss weights: {self.loss_weights}")
        
        self.opt_A = optim.Adam(self.unet_A.parameters(), 
                               lr=config.get('learning_rate', 0.001))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_A, 
            patience=config.get('scheduler_patience', 5),
            factor=config.get('scheduler_factor', 0.5)
        )
        
        # Criterio
        self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.experiment_name = config.get('experiment_name', 
                                        f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
    
    def _load_checkpoints(self, signal_to_image_path, image_to_signal_path):
        """Load and validate checkpoints for both UNets"""
        
        def load_checkpoint(path, model):
            checkpoint = torch.load(path, map_location=self.device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        print(f"Loading constructive model")                
        # Load UNet A (signal -> image)
        load_checkpoint(signal_to_image_path, self.unet_A)
        
        print(f"Loading supervised model")
        # Load UNet B (image -> signal) and freeze
        load_checkpoint(image_to_signal_path, self.unet_B)
        for param in self.unet_B.parameters():
            param.requires_grad = False
        self.unet_B.eval()
            
    def train(self, train_loader, val_loader, num_epochs=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f'training_results_{self.experiment_name}_{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(f'runs/photoacoustic_supervised_{timestamp}')
        
        best_val_loss = float('inf')
        history = {
            'train_loss': [], 'val_loss': [],
            'loss_direct': [], 'loss_physical': [],
            'loss_struct': [], 'similarity_penalty': []
        }
        
        for epoch in range(num_epochs):
            # Modo entrenamiento para UNet A
            self.unet_A.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as t:
                for signal, image in t:
                    signal, image = signal.to(self.device), image.to(self.device)
                    
                    # Zero gradients
                    self.opt_A.zero_grad()
                    
                    # Forward pass
                    reconstructed_image = self.unet_A(signal)
                    predicted_signal = self.unet_B(reconstructed_image)
                    
                    # Pérdida directa: qué tan bien reconstruye la imagen
                    loss_direct = self.criterion(reconstructed_image, image)
                   
                    # Pérdida física: la señal que genera la imagen reconstruida 
                    # debe parecerse a la señal original
                    loss_physical = self.criterion(predicted_signal, signal)
                     
                    # Pérdida estructural
                    loss_struct = calculate_structural_loss(reconstructed_image, image)
                    
                    # Penalización por similitud con entrada
                    similarity_penalty = calculate_similarity_penalty(reconstructed_image, signal)
                    
                    # Pérdida total
                    total_loss = (self.loss_weights['direct'] * loss_direct + 
                        self.loss_weights['physical'] * loss_physical +
                        self.loss_weights['struct'] * loss_struct +
                        self.loss_weights['similarity'] * similarity_penalty)
                    
                    # Backward y optimización
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.unet_A.parameters(), max_norm=1.0)

                    self.opt_A.step()
                    
                    train_loss += total_loss.item()
                    
                    # Actualizar barra de progreso
                    t.set_postfix({
                        'loss': total_loss.item(),
                        'loss_direct': loss_direct.item(),
                        'loss_physical': loss_physical.item(),
                        'loss_struct': loss_struct.item(),
                        'similarity_penalty': similarity_penalty.item(),
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
            writer.add_scalar('Loss/structural', val_metrics['loss_struct'], epoch)
            writer.add_scalar('Penalty/similarity', val_metrics['similarity_penalty'], epoch)
            
            # Guardar histórico
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['loss_direct'].append(val_metrics['loss_direct'])
            history['loss_physical'].append(val_metrics['loss_physical'])
            history['loss_struct'].append(val_metrics['loss_struct'])
            history['similarity_penalty'].append(val_metrics['similarity_penalty'])
            
            # Visualización periódica
            if (epoch + 1) % 10 == 0:
                self.visualize_results(val_loader, epoch, save_dir)
        
        self.save_training_history(history, save_dir)
        return history
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.unet_A.eval()
        total_loss = 0
        metrics = {'loss_direct': 0, 'loss_physical': 0, 'loss_struct': 0, 'similarity_penalty': 0}
        
        for signal, image in val_loader:
            signal, image = signal.to(self.device), image.to(self.device)
            
            # Forward passes
            reconstructed_image = self.unet_A(signal)
            predicted_signal = self.unet_B(reconstructed_image)
            
            # Calcular pérdidas
            loss_direct = self.criterion(reconstructed_image, image)
            loss_physical = self.criterion(predicted_signal, signal)
            loss_struct = calculate_structural_loss(reconstructed_image, image)
            similarity_penalty = calculate_similarity_penalty(reconstructed_image, signal)
            
            total_loss = (self.loss_weights['direct'] * loss_direct + 
                        self.loss_weights['physical'] * loss_physical +
                        self.loss_weights['struct'] * loss_struct +
                        self.loss_weights['similarity'] * similarity_penalty)
            
            metrics['loss_direct'] += loss_direct.item()
            metrics['loss_physical'] += loss_physical.item()
            metrics['loss_struct'] += loss_struct.item()
            metrics['similarity_penalty'] += similarity_penalty.item()
        
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
            
            # Calcular pérdidas
            loss_direct = self.criterion(reconstructed_image, image).item()
            loss_physical = self.criterion(predicted_signal, signal).item()
            loss_struct = calculate_structural_loss(reconstructed_image, image).item()
            similarity = calculate_similarity_penalty(reconstructed_image, signal).item()
            
            plt.figure(figsize=(15, 5))
            
            # Señal original
            plt.subplot(141)
            plt.imshow(signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title('Original Signal')
            plt.colorbar()
            
            # Imagen objetivo
            plt.subplot(142)
            plt.imshow(image.cpu().numpy()[0, 0], cmap='gray')
            plt.title(f'Target Image\nDirect Loss: {loss_direct:.4f}\nStruct Loss: {loss_struct:.4f}')
            plt.colorbar()
            
            # Imagen reconstruida
            plt.subplot(143)
            plt.imshow(reconstructed_image.cpu().numpy()[0, 0], cmap='gray')
            plt.title(f'Reconstructed Image\nSimilarity Penalty: {similarity:.4f}')
            plt.colorbar()
            
            # Señal predicha
            plt.subplot(144)
            plt.imshow(predicted_signal.cpu().numpy()[0, 0], cmap='viridis')
            plt.title(f'Physical Validation\nPhysical Loss: {loss_physical:.4f}')
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/epoch_{epoch+1}_results.png')
            plt.close()
        
    def save_training_history(self, history, save_dir):
        # Move data to CPU and convert to numpy
        cpu_history = {}
        for key, value in history.items():
            if isinstance(value, list):
                # If the value is a list of tensors
                if len(value) > 0 and torch.is_tensor(value[0]):
                    cpu_history[key] = np.array([tensor.cpu().detach().numpy() for tensor in value])
                else:
                    cpu_history[key] = np.array(value)
            elif torch.is_tensor(value):
                # If the value is a single tensor
                cpu_history[key] = value.cpu().detach().numpy()
            else:
                cpu_history[key] = np.array(value)
        
        # Save numpy arrays
        for key, value in cpu_history.items():
            np.save(f'{save_dir}/{key}.npy', value)
        
        # Plot with CPU data
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(cpu_history['train_loss'], label='Train Loss')
        plt.plot(cpu_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        
        plt.subplot(122)
        plt.plot(cpu_history['loss_direct'], label='Direct Loss')
        plt.plot(cpu_history['loss_physical'], label='Physical Loss')
        plt.plot(cpu_history['loss_struct'], label='Structural Loss')
        plt.plot(cpu_history['similarity_penalty'], label='Similarity Penalty')
        plt.xlabel('Epoch')
        plt.ylabel('Component Losses') 
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config JSON')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Establecer semilla global
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    
    with open(args.config) as f:
         config = json.load(f)
    
    # Guardar la configuración y la semilla usada
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_base_dir = f'ablation_studies_{timestamp}'
    os.makedirs(experiment_base_dir, exist_ok=True)
    
    # Guardar configuración con la semilla utilizada
    config['seed'] = args.seed
    with open(f'{experiment_base_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Cargar datos
    train_loader, val_loader, test_loader = load_and_preprocess_data(
        config['data_dir'], 
        batch_size=config['batch_size'],
        seed=args.seed  # Pasar la semilla al data loader
    )
    
    # Inicializar y entrenar
    for experiment_config in config['ablation_studies']:
        print(f"\nStarting experiment: {experiment_config['experiment_name']}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Crear configuración específica
        current_config = experiment_config.copy()
        current_config.update(config['training'])
        current_config['device'] = device
        current_config['seed'] = args.seed
        
        # Crear nuevo trainer
        trainer = SupervisedUNetTrainer(current_config)
        
        # Entrenar
        history = trainer.train(
            train_loader, 
            val_loader, 
            config['training']['num_epochs']
        )
        
        # Liberar memoria
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()