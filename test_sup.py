import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
from preprocess.preprocess_simulated_data import load_and_preprocess_data
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

def evaluate_supervised_model(unet_A, unet_B, test_loader, device, save_dir='test_results_supervised'):
    """
    Evalúa el modelo supervisado en el conjunto de test
    """
    os.makedirs(save_dir, exist_ok=True)
    unet_A.eval()
    unet_B.eval()
    
    # Métricas para reconstrucción y física
    metrics = {
        'mse_scores': [],
        'mae_scores': [],
        'ssim_scores': [],
        'physical_consistency': [],  # MSE entre señal original y predicha
        'correlation_scores': []     # Correlación entre señales
    }
    
    print("Evaluando el modelo en el conjunto de test...")
    with torch.no_grad():
        for i, (signal, image) in enumerate(tqdm(test_loader)):
            signal, image = signal.to(device), image.to(device)
            
            # Forward passes
            reconstructed_image = unet_A(signal)
            predicted_signal = unet_B(reconstructed_image)
            
            # Métricas de reconstrucción de imagen
            mse = mean_squared_error(image.cpu().numpy().flatten(), 
                                   reconstructed_image.cpu().numpy().flatten())
            mae = mean_absolute_error(image.cpu().numpy().flatten(), 
                                    reconstructed_image.cpu().numpy().flatten())
            
            # SSIM para calidad estructural
            ssim_score = ssim(image.cpu().numpy()[0, 0],
                            reconstructed_image.cpu().numpy()[0, 0],
                            data_range=image.cpu().numpy().max() - image.cpu().numpy().min())
            
            # Consistencia física
            physical_mse = mean_squared_error(signal.cpu().numpy().flatten(),
                                           predicted_signal.cpu().numpy().flatten())
            
            # Correlación entre señales
            correlation = np.corrcoef(signal.cpu().numpy().flatten(),
                                    predicted_signal.cpu().numpy().flatten())[0,1]
            
            # Guardar métricas
            metrics['mse_scores'].append(mse)
            metrics['mae_scores'].append(mae)
            metrics['ssim_scores'].append(ssim_score)
            metrics['physical_consistency'].append(physical_mse)
            metrics['correlation_scores'].append(correlation)
            
            # Guardar visualizaciones
            if i < 5:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Señal original
                im0 = axes[0].imshow(signal.cpu().numpy()[0, 0], cmap='viridis')
                axes[0].set_title('Original Signal')
                plt.colorbar(im0, ax=axes[0])
                
                # Imagen objetivo
                im1 = axes[1].imshow(image.cpu().numpy()[0, 0], cmap='gray')
                axes[1].set_title(f'Target Image\nMSE: {mse:.4f}\nSSIM: {ssim_score:.4f}')
                plt.colorbar(im1, ax=axes[1])
                
                # Imagen reconstruida
                im2 = axes[2].imshow(reconstructed_image.cpu().numpy()[0, 0], cmap='gray')
                axes[2].set_title('Reconstructed Image')
                plt.colorbar(im2, ax=axes[2])
                
                # Señal predicha
                im3 = axes[3].imshow(predicted_signal.cpu().numpy()[0, 0], cmap='viridis')
                axes[3].set_title(f'Physical Validation\nMSE: {physical_mse:.4f}\nCorr: {correlation:.4f}')
                plt.colorbar(im3, ax=axes[3])
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/test_sample_{i}.png')
                plt.close()
    
    # Calcular métricas promedio
    avg_metrics = {}
    std_metrics = {}
    
    for key in metrics:
        avg_metrics[key] = np.mean(metrics[key])
        std_metrics[key] = np.std(metrics[key])
    
    # Guardar métricas en archivo
    with open(f'{save_dir}/metrics.txt', 'w') as f:
        f.write("Métricas de Evaluación:\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Reconstrucción de Imagen:\n")
        f.write(f"MSE: {avg_metrics['mse_scores']:.6f} ± {std_metrics['mse_scores']:.6f}\n")
        f.write(f"MAE: {avg_metrics['mae_scores']:.6f} ± {std_metrics['mae_scores']:.6f}\n")
        f.write(f"SSIM: {avg_metrics['ssim_scores']:.6f} ± {std_metrics['ssim_scores']:.6f}\n\n")
        
        f.write("Consistencia Física:\n")
        f.write(f"MSE Señal: {avg_metrics['physical_consistency']:.6f} ± {std_metrics['physical_consistency']:.6f}\n")
        f.write(f"Correlación: {avg_metrics['correlation_scores']:.6f} ± {std_metrics['correlation_scores']:.6f}\n")
    
    # Crear histogramas
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.histplot(metrics['mse_scores'], bins=30)
    plt.title('MSE Distribution (Image)')
    
    plt.subplot(132)
    sns.histplot(metrics['ssim_scores'], bins=30)
    plt.title('SSIM Distribution')
    
    plt.subplot(133)
    sns.histplot(metrics['correlation_scores'], bins=30)
    plt.title('Signal Correlation Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png')
    plt.close()
    
    # Scatter plot de SSIM vs Physical Consistency
    plt.figure(figsize=(8, 8))
    plt.scatter(metrics['ssim_scores'], metrics['correlation_scores'], alpha=0.5)
    plt.xlabel('SSIM (Image Quality)')
    plt.ylabel('Signal Correlation')
    plt.title('Image Quality vs Physical Consistency')
    plt.savefig(f'{save_dir}/quality_vs_physics.png')
    plt.close()
    
    return avg_metrics, std_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate supervised model performance.")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the supervised model checkpoint")
    parser.add_argument('--supervisor_path', type=str, required=True,
                       help="Path to the supervisor (UNet B) checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Crear modelos
    unet_A = UNet(in_channels=1, out_channels=1).to(device)
    unet_B = UNet(in_channels=1, out_channels=1).to(device)
    
    # Cargar checkpoints
    print(f"Cargando modelo supervisado desde: {args.model_path}")
    checkpoint_A = torch.load(args.model_path, map_location=device)
    if 'unet_A_state_dict' in checkpoint_A:
        unet_A.load_state_dict(checkpoint_A['unet_A_state_dict'])
    else:
        unet_A.load_state_dict(checkpoint_A)
    
    print(f"Cargando supervisor desde: {args.supervisor_path}")
    checkpoint_B = torch.load(args.supervisor_path, map_location=device)
    if 'state_dict' in checkpoint_B:
        unet_B.load_state_dict(checkpoint_B['state_dict'])
    else:
        unet_B.load_state_dict(checkpoint_B)
    
    # Cargar datos de test
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Evaluar modelo
    avg_metrics, std_metrics = evaluate_supervised_model(unet_A, unet_B, test_loader, device)
    
    print("\nResultados de la evaluación:")
    print("\nReconstrucción de Imagen:")
    print(f"MSE: {avg_metrics['mse_scores']:.6f} ± {std_metrics['mse_scores']:.6f}")
    print(f"MAE: {avg_metrics['mae_scores']:.6f} ± {std_metrics['mae_scores']:.6f}")
    print(f"SSIM: {avg_metrics['ssim_scores']:.6f} ± {std_metrics['ssim_scores']:.6f}")
    
    print("\nConsistencia Física:")
    print(f"MSE Señal: {avg_metrics['physical_consistency']:.6f} ± {std_metrics['physical_consistency']:.6f}")
    print(f"Correlación: {avg_metrics['correlation_scores']:.6f} ± {std_metrics['correlation_scores']:.6f}")

if __name__ == "__main__":
    main()