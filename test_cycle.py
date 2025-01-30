import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
from preprocess.preprocess_simulated_data import load_and_preprocess_data
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def evaluate_cycle_model(unet_A, unet_B, test_loader, device, save_dir='test_results_cycle'):
    """
    Evalúa el modelo cíclico en el conjunto de test
    """
    os.makedirs(save_dir, exist_ok=True)
    unet_A.eval()
    unet_B.eval()
    
    # Métricas para señal->imagen->señal
    metrics_A = {
        'mse_image': [], 'mae_image': [],  # Reconstrucción de imagen
        'mse_cycle': [], 'mae_cycle': [],  # Consistencia cíclica
        'physical_consistency': []          # Consistencia física
    }
    
    print("Evaluando el modelo en el conjunto de test...")
    with torch.no_grad():
        for i, (signal, image) in enumerate(tqdm(test_loader)):
            signal, image = signal.to(device), image.to(device)
            
            # Forward passes
            reconstructed_image = unet_A(signal)
            predicted_signal = unet_B(reconstructed_image)
            
            # Calcular métricas para reconstrucción de imagen
            mse_image = mean_squared_error(
                image.cpu().numpy().flatten(),
                reconstructed_image.cpu().numpy().flatten()
            )
            mae_image = mean_absolute_error(
                image.cpu().numpy().flatten(),
                reconstructed_image.cpu().numpy().flatten()
            )
            
            # Calcular métricas para consistencia cíclica
            mse_cycle = mean_squared_error(
                signal.cpu().numpy().flatten(),
                predicted_signal.cpu().numpy().flatten()
            )
            mae_cycle = mean_absolute_error(
                signal.cpu().numpy().flatten(),
                predicted_signal.cpu().numpy().flatten()
            )
            
            # Consistencia física (correlación entre señales)
            physical_consistency = np.corrcoef(
                signal.cpu().numpy().flatten(),
                predicted_signal.cpu().numpy().flatten()
            )[0,1]
            
            # Guardar métricas
            metrics_A['mse_image'].append(mse_image)
            metrics_A['mae_image'].append(mae_image)
            metrics_A['mse_cycle'].append(mse_cycle)
            metrics_A['mae_cycle'].append(mae_cycle)
            metrics_A['physical_consistency'].append(physical_consistency)
            
            # Guardar visualizaciones
            if i < 5:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Señal original
                im0 = axes[0].imshow(signal[0, 0].cpu().numpy(), cmap='viridis')
                axes[0].set_title('Original Signal')
                plt.colorbar(im0, ax=axes[0])
                
                # Imagen objetivo
                im1 = axes[1].imshow(image[0, 0].cpu().numpy(), cmap='gray')
                axes[1].set_title(f'Target Image\nMSE: {mse_image:.4f}')
                plt.colorbar(im1, ax=axes[1])
                
                # Imagen reconstruida
                im2 = axes[2].imshow(reconstructed_image[0, 0].cpu().numpy(), cmap='gray')
                axes[2].set_title('Reconstructed Image')
                plt.colorbar(im2, ax=axes[2])
                
                # Señal predicha
                im3 = axes[3].imshow(predicted_signal[0, 0].cpu().numpy(), cmap='viridis')
                axes[3].set_title(f'Cycle Signal\nMSE: {mse_cycle:.4f}')
                plt.colorbar(im3, ax=axes[3])
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/test_sample_{i}.png')
                plt.close()
    
    # Calcular métricas promedio
    avg_metrics = {}
    std_metrics = {}
    
    for key in metrics_A:
        avg_metrics[key] = np.mean(metrics_A[key])
        std_metrics[key] = np.std(metrics_A[key])
    
    # Guardar métricas en archivo
    with open(f'{save_dir}/metrics.txt', 'w') as f:
        f.write("Métricas de Evaluación:\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Reconstrucción de Imagen:\n")
        f.write(f"MSE: {avg_metrics['mse_image']:.6f} ± {std_metrics['mse_image']:.6f}\n")
        f.write(f"MAE: {avg_metrics['mae_image']:.6f} ± {std_metrics['mae_image']:.6f}\n\n")
        
        f.write("Consistencia Cíclica:\n")
        f.write(f"MSE: {avg_metrics['mse_cycle']:.6f} ± {std_metrics['mse_cycle']:.6f}\n")
        f.write(f"MAE: {avg_metrics['mae_cycle']:.6f} ± {std_metrics['mae_cycle']:.6f}\n\n")
        
        f.write("Consistencia Física:\n")
        f.write(f"Correlación: {avg_metrics['physical_consistency']:.6f} ± {std_metrics['physical_consistency']:.6f}\n")
    
    # Crear histogramas
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.histplot(metrics_A['mse_image'], bins=30)
    plt.title('MSE Distribution (Image)')
    
    plt.subplot(132)
    sns.histplot(metrics_A['mse_cycle'], bins=30)
    plt.title('MSE Distribution (Cycle)')
    
    plt.subplot(133)
    sns.histplot(metrics_A['physical_consistency'], bins=30)
    plt.title('Physical Consistency Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png')
    plt.close()
    
    return avg_metrics, std_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate cycle model performance.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help="Path to the cycle model checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Crear modelos
    unet_A = UNet(in_channels=1, out_channels=1).to(device)
    unet_B = UNet(in_channels=1, out_channels=1).to(device)
    
    # Cargar checkpoint
    print(f"Cargando modelo desde: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    unet_A.load_state_dict(checkpoint['unet_A_state_dict'])
    unet_B.load_state_dict(checkpoint['unet_B_state_dict'])
    
    # Cargar datos de test
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Evaluar modelo
    avg_metrics, std_metrics = evaluate_cycle_model(unet_A, unet_B, test_loader, device)
    
    print("\nResultados de la evaluación:")
    print("\nReconstrucción de Imagen:")
    print(f"MSE: {avg_metrics['mse_image']:.6f} ± {std_metrics['mse_image']:.6f}")
    print(f"MAE: {avg_metrics['mae_image']:.6f} ± {std_metrics['mae_image']:.6f}")
    
    print("\nConsistencia Cíclica:")
    print(f"MSE: {avg_metrics['mse_cycle']:.6f} ± {std_metrics['mse_cycle']:.6f}")
    print(f"MAE: {avg_metrics['mae_cycle']:.6f} ± {std_metrics['mae_cycle']:.6f}")
    
    print("\nConsistencia Física:")
    print(f"Correlación: {avg_metrics['physical_consistency']:.6f} ± {std_metrics['physical_consistency']:.6f}")

if __name__ == "__main__":
    main()