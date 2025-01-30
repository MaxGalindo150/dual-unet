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

def evaluate_inverse_model(model, test_loader, device, save_dir='test_results_inverse'):
    """
    Evalúa el modelo inverso (imagen → señal) en el conjunto de test
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Listas para almacenar métricas
    mse_scores = []
    mae_scores = []
    predictions = []
    targets = []
    
    print("Evaluando el modelo inverso en el conjunto de test...")
    with torch.no_grad():
        for i, (signal, image) in enumerate(tqdm(test_loader)):
            # Nota: cambiamos el orden porque ahora la imagen es la entrada
            image, signal = image.to(device), signal.to(device)
            predicted_signal = model(image)
            
            # Guardar predicciones y targets
            predictions.extend(predicted_signal.cpu().numpy())
            targets.extend(signal.cpu().numpy())
            
            # Calcular métricas por batch
            mse = mean_squared_error(signal.cpu().numpy().flatten(), 
                                   predicted_signal.cpu().numpy().flatten())
            mae = mean_absolute_error(signal.cpu().numpy().flatten(), 
                                    predicted_signal.cpu().numpy().flatten())
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            
            # Guardar algunas visualizaciones
            if i < 5:  # Guardar solo las primeras 5 muestras
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Imagen de entrada
                im0 = axes[0].imshow(image[0, 0].cpu().numpy(), cmap='gray')
                axes[0].set_title('Input Image')
                plt.colorbar(im0, ax=axes[0])
                
                # Señal objetivo
                im1 = axes[1].imshow(signal[0, 0].cpu().numpy(), cmap='viridis')
                axes[1].set_title('Ground Truth Signal')
                plt.colorbar(im1, ax=axes[1])
                
                # Señal predicha
                im2 = axes[2].imshow(predicted_signal[0, 0].cpu().numpy(), cmap='viridis')
                axes[2].set_title('Predicted Signal')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(f'{save_dir}/test_sample_{i}.png')
                plt.close()
    
    # Calcular métricas promedio
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    std_mse = np.std(mse_scores)
    std_mae = np.std(mae_scores)
    
    # Guardar métricas en un archivo
    with open(f'{save_dir}/metrics.txt', 'w') as f:
        f.write(f'Average MSE: {avg_mse:.6f} ± {std_mse:.6f}\n')
        f.write(f'Average MAE: {avg_mae:.6f} ± {std_mae:.6f}\n')
    
    # Crear histograma de errores
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(mse_scores, bins=30)
    plt.title('Distribution of MSE')
    plt.xlabel('MSE')
    
    plt.subplot(1, 2, 2)
    sns.histplot(mae_scores, bins=30)
    plt.title('Distribution of MAE')
    plt.xlabel('MAE')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_distribution.png')
    plt.close()
    
    # Crear scatter plot de valores predichos vs reales
    plt.figure(figsize=(8, 8))
    plt.scatter(np.array(targets).flatten(), 
               np.array(predictions).flatten(), 
               alpha=0.1)
    plt.plot([np.min(targets), np.max(targets)], 
             [np.min(targets), np.max(targets)], 'r--')  # Línea de identidad
    plt.xlabel('Ground Truth Signal Values')
    plt.ylabel('Predicted Signal Values')
    plt.title('Predicted vs Ground Truth Signal Values')
    plt.savefig(f'{save_dir}/prediction_scatter.png')
    plt.close()
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'mse_std': std_mse,
        'mae_std': std_mae
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the inverse U-Net model (image to signal).")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the trained inverse model weights")
    args = parser.parse_args()
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar el modelo
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Cargar los pesos del modelo
    print(f"Cargando modelo desde: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    
    # Cargar datos de test
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Evaluar modelo
    metrics = evaluate_inverse_model(model, test_loader, device)
    
    print("\nResultados de la evaluación del modelo inverso:")
    print(f"MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"MAE: {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")

if __name__ == "__main__":
    main()