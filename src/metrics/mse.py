import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from models.unet_model import UNet
from preprocess.preprocess_simulated_data import load_and_preprocess_data

def calculate_batch_mse(model, data_loader, device, save_dir, model_name):
    """
    Calcula el MSE para cada imagen en el conjunto de datos y guarda los resultados.
    
    Args:
        model: Modelo de reconstrucción
        data_loader: DataLoader con datos de prueba
        device: Dispositivo para computación
        save_dir: Directorio para guardar resultados
        model_name: Nombre identificador del modelo
    
    Returns:
        numpy.ndarray: Array con todos los valores MSE calculados
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Lista para almacenar resultados
    mse_values = []
    sample_indices = []  # Para mantener registro del índice de cada muestra
    
    model.eval()
    print(f"\nCalculando MSE para el modelo: {model_name}")
    
    with torch.no_grad():
        for batch_idx, (signal, target) in enumerate(tqdm(data_loader)):
            signal, target = signal.to(device), target.to(device)
            
            # Generar predicción
            prediction = model(signal)
            
            # Calcular MSE para cada imagen en el batch
            for i in range(signal.size(0)):
                target_np = target[i, 0].cpu().numpy()
                pred_np = prediction[i, 0].cpu().numpy()
                
                # Calcular MSE
                mse = mean_squared_error(target_np, pred_np)
                
                # Guardar resultados
                mse_values.append(mse)
                sample_indices.append(batch_idx * data_loader.batch_size + i)
                
                # Guardar algunas visualizaciones de ejemplo
                if len(mse_values) <= 5:  # Primeras 5 muestras
                    save_sample_visualization(
                        signal[i, 0].cpu().numpy(),
                        target_np,
                        pred_np,
                        mse,
                        len(mse_values),
                        save_dir,
                        model_name
                    )
    
    # Convertir a array numpy
    mse_values = np.array(mse_values)
    
    # Crear DataFrame con los resultados
    results_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'mse': mse_values
    })
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(save_dir / f'{model_name}_mse_values_{timestamp}.csv', index=False)
    
    # Crear y guardar histograma de MSE
    plt.figure(figsize=(10, 6))
    sns.histplot(data=mse_values, bins=30)
    plt.title(f'Distribución de MSE - {model_name}')
    plt.xlabel('MSE')
    plt.ylabel('Frecuencia')
    
    # Añadir líneas verticales para estadísticas importantes
    plt.axvline(np.mean(mse_values), color='r', linestyle='--', 
                label=f'Media: {np.mean(mse_values):.6f}')
    plt.axvline(np.median(mse_values), color='g', linestyle='--', 
                label=f'Mediana: {np.median(mse_values):.6f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_mse_distribution_{timestamp}.png')
    plt.close()
    
    # Guardar estadísticas básicas
    stats = {
        'mean': np.mean(mse_values),
        'median': np.median(mse_values),
        'std': np.std(mse_values),
        'min': np.min(mse_values),
        'max': np.max(mse_values)
    }
    
    # Guardar estadísticas en formato legible
    with open(save_dir / f'{model_name}_mse_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Estadísticas MSE para {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Media: {stats['mean']:.6f}\n")
        f.write(f"Mediana: {stats['median']:.6f}\n")
        f.write(f"Desviación Estándar: {stats['std']:.6f}\n")
        f.write(f"Mínimo: {stats['min']:.6f}\n")
        f.write(f"Máximo: {stats['max']:.6f}\n")
    
    return mse_values

def save_sample_visualization(signal, target, prediction, mse, index, save_dir, model_name):
    """
    Guarda visualización de una muestra individual con su MSE.
    
    Args:
        signal: Señal de entrada
        target: Imagen objetivo
        prediction: Imagen predicha
        mse: Valor MSE calculado
        index: Índice de la muestra
        save_dir: Directorio para guardar
        model_name: Nombre del modelo
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Señal original
    im0 = axes[0].imshow(signal, cmap='viridis')
    axes[0].set_title('Input Signal')
    plt.colorbar(im0, ax=axes[0])
    
    # Imagen objetivo
    im1 = axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Target Image')
    plt.colorbar(im1, ax=axes[1])
    
    # Imagen reconstruida
    im2 = axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(f'Reconstructed Image\nMSE: {mse:.6f}')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_sample_{index}_mse_{mse:.6f}.png')
    plt.close()

