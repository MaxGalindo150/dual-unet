import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

from models.unet_model import UNet
from preprocess.preprocess_simulated_data import load_and_preprocess_data

def calculate_psnr(target, prediction, data_range=None):
    """
    Calcula el Peak Signal-to-Noise Ratio (PSNR) entre la imagen objetivo y la predicha.
    
    El PSNR se define como:
    PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    donde MAX es el valor máximo posible del pixel y MSE es el error cuadrático medio.
    
    Args:
        target: Imagen objetivo (numpy array)
        prediction: Imagen predicha (numpy array)
        data_range: Rango dinámico de las imágenes (max-min). 
                   Si es None, se calcula del target.
    
    Returns:
        float: Valor PSNR en dB
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    # Asegurar que los datos están en el mismo formato
    target = target.astype(np.float32)
    prediction = prediction.astype(np.float32)
    
    # Calcular MSE
    mse = np.mean((target - prediction) ** 2)
    
    # Evitar división por cero o log de cero
    if mse == 0:
        return float('inf')
    
    # Calcular PSNR
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    
    return psnr

def evaluate_psnr(model, test_loader, device, save_dir='psnr_evaluation'):
    """
    Evalúa el modelo usando PSNR y genera visualizaciones relevantes.
    
    Args:
        model: Modelo de reconstrucción
        test_loader: DataLoader con datos de prueba
        device: Dispositivo para computación (CPU/GPU)
        save_dir: Directorio para guardar resultados
    
    Returns:
        dict: Estadísticas de PSNR (media, std, min, max)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    psnr_values = []
    
    print("Evaluando PSNR en el conjunto de test...")
    with torch.no_grad():
        for i, (signal, target) in enumerate(test_loader):
            signal, target = signal.to(device), target.to(device)
            
            # Generar predicción
            prediction = model(signal)
            
            # Convertir a numpy para cálculos
            target_np = target.cpu().numpy()[0, 0]
            pred_np = prediction.cpu().numpy()[0, 0]
            
            # Calcular PSNR
            psnr = calculate_psnr(target_np, pred_np)
            psnr_values.append(psnr)
            
            # Guardar algunas muestras con visualización
            if i < 5:  # Guardar las primeras 5 muestras
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Señal original (entrada)
                im0 = axes[0].imshow(signal.cpu().numpy()[0, 0], cmap='viridis')
                axes[0].set_title('Input Signal')
                plt.colorbar(im0, ax=axes[0])
                
                # Imagen objetivo
                im1 = axes[1].imshow(target_np, cmap='gray')
                axes[1].set_title('Target Image')
                plt.colorbar(im1, ax=axes[1])
                
                # Imagen reconstruida
                im2 = axes[2].imshow(pred_np, cmap='gray')
                axes[2].set_title(f'Reconstructed Image\nPSNR: {psnr:.2f} dB')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(save_dir / f'sample_{i}_psnr_{psnr:.2f}.png')
                plt.close()
    
    # Calcular estadísticas
    psnr_values = np.array(psnr_values)
    stats = {
        'mean_psnr': float(np.mean(psnr_values)),
        'std_psnr': float(np.std(psnr_values)),
        'min_psnr': float(np.min(psnr_values)),
        'max_psnr': float(np.max(psnr_values)),
        'median_psnr': float(np.median(psnr_values))
    }
    
    # Crear visualización de la distribución de PSNR
    plt.figure(figsize=(10, 6))
    sns.histplot(psnr_values, bins=30)
    plt.title('Distribution of PSNR Values')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    
    # Agregar líneas verticales para estadísticas importantes
    plt.axvline(stats['mean_psnr'], color='r', linestyle='--', 
                label=f'Mean: {stats["mean_psnr"]:.2f} dB')
    plt.axvline(stats['median_psnr'], color='g', linestyle='--', 
                label=f'Median: {stats["median_psnr"]:.2f} dB')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'psnr_distribution.png')
    plt.close()
    
    # Guardar estadísticas en un archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(save_dir / f'psnr_stats_{timestamp}.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    # También guardar en formato legible
    with open(save_dir / f'psnr_report_{timestamp}.txt', 'w') as f:
        f.write("PSNR Evaluation Results\n")
        f.write("=====================\n\n")
        f.write(f"Mean PSNR: {stats['mean_psnr']:.2f} dB\n")
        f.write(f"Median PSNR: {stats['median_psnr']:.2f} dB\n")
        f.write(f"Standard Deviation: {stats['std_psnr']:.2f} dB\n")
        f.write(f"Range: {stats['min_psnr']:.2f} - {stats['max_psnr']:.2f} dB\n")
    
    print("\nResultados del PSNR:")
    print(f"Media: {stats['mean_psnr']:.2f} dB")
    print(f"Mediana: {stats['median_psnr']:.2f} dB")
    print(f"Desviación Estándar: {stats['std_psnr']:.2f} dB")
    print(f"Rango: {stats['min_psnr']:.2f} - {stats['max_psnr']:.2f} dB")
    
    return stats

def main():
    """
    Función principal para ejecutar la evaluación PSNR
    
    Uso:
    python evaluate_psnr.py --model_path path/to/model --save_dir results/psnr
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate PSNR for photoacoustic reconstruction")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument('--save_dir', type=str, default='psnr_results',
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar modelo (ajusta según tu arquitectura específica)
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Cargar datos (ajusta según tu pipeline de datos)
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Ejecutar evaluación
    stats = evaluate_psnr(model, test_loader, device, args.save_dir)

if __name__ == "__main__":
    main()