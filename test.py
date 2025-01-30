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
from physics_informed import SA_UNet
from attention_unet_model import AttentionUNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def evaluate_model(model, test_loader, device, save_dir='test_results'):
    """
    Evalúa el modelo en el conjunto de test y guarda las métricas y visualizaciones
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Listas para almacenar métricas
    mse_scores = []
    mae_scores = []
    predictions = []
    targets = []
    
    print("Evaluando el modelo en el conjunto de test...")
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Guardar predicciones y targets
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
            # Calcular métricas por batch
            mse = mean_squared_error(target.cpu().numpy().flatten(), 
                                   output.cpu().numpy().flatten())
            mae = mean_absolute_error(target.cpu().numpy().flatten(), 
                                    output.cpu().numpy().flatten())
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            
            # Guardar algunas visualizaciones
            if i < 5:  # Guardar solo las primeras 5 muestras
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(data[0, 0].cpu().numpy())
                axes[0].set_title('Input')
                axes[1].imshow(target[0, 0].cpu().numpy())
                axes[1].set_title('Ground Truth')
                axes[2].imshow(output[0, 0].cpu().numpy())
                axes[2].set_title('Prediction')
                
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
    plt.plot([0, 1], [0, 1], 'r--')  # Línea de identidad
    plt.xlabel('Ground Truth Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Ground Truth Values')
    plt.savefig(f'{save_dir}/prediction_scatter.png')
    plt.close()
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'mse_std': std_mse,
        'mae_std': std_mae
    }

def main():
    
    parser = argparse.ArgumentParser(description="Evaluate a U-Net model for photoacoustic image reconstruction.")
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'attention_unet', 'sa_unet'],)
    args = parser.parse_args()
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar el modelo
    
    if args.model_name == 'unet':
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif args.model_name == 'attention_unet':
        model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    elif args.model_name == 'sa_unet':
        model = SA_UNet().to(device)
    
    # Cargar los pesos del mejor modelo
    # Encuentra el directorio de resultados más reciente
    training_dirs = [d for d in os.listdir() if d.startswith(f"training_results_{args.model_name}_")]
    print(training_dirs)
    latest_training_dir = max(training_dirs, key=lambda x: os.path.getctime(x))
    model_path = os.path.join(latest_training_dir, 'best_model.pth')
    
    print(f"Cargando modelo desde: {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    # Cargar datos de test
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Evaluar modelo
    metrics = evaluate_model(model, test_loader, device)
    
    print("\nResultados de la evaluación:")
    print(f"MSE: {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"MAE: {metrics['mae']:.6f} ± {metrics['mae_std']:.6f}")

if __name__ == "__main__":
    main()