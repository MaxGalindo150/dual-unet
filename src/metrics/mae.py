import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

def create_enhanced_mae_histogram(mae_values, model_name, save_dir, timestamp):
    """
    Crea un histograma mejorado de la distribución de MAE con elementos visuales adicionales.
    El MAE representa el promedio de las diferencias absolutas entre las predicciones y los valores reales,
    proporcionando una medida de error en las mismas unidades que los datos originales.
    """
    # Configurar el estilo general para una visualización profesional
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear una figura con dimensiones optimizadas para la visualización
    fig = plt.figure(figsize=(14, 8))
    
    # Crear el grid para el histograma
    gs = plt.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    # Definir una paleta de colores profesional y accesible
    colors = {
        'hist': '#3498db',      # Azul para el histograma
        'mean': '#e74c3c',      # Rojo para la media
        'median': '#2ecc71',    # Verde para la mediana
        'kde': '#2c3e50',       # Azul oscuro para KDE
        'text_box': '#f8f9fa'   # Gris claro para el fondo del texto
    }
    
    # Crear el histograma principal con densidad para mejor interpretación
    sns.histplot(data=mae_values, 
                bins=30,
                color=colors['hist'],
                alpha=0.7,
                stat='density',
                ax=ax,
                label='Histograma')
    
    # Añadir la curva de densidad KDE para visualizar la distribución
    sns.kdeplot(data=mae_values,
                color=colors['kde'],
                linewidth=2,
                ax=ax,
                label='Densidad KDE')
    
    # Calcular estadísticas descriptivas
    mean_val = np.mean(mae_values)
    median_val = np.median(mae_values)
    std_val = np.std(mae_values)
    
    # Añadir líneas verticales para media y mediana
    ax.axvline(mean_val, color=colors['mean'], linestyle='--', linewidth=2,
               label=f'Media: {mean_val:.6f}')
    ax.axvline(median_val, color=colors['median'], linestyle='--', linewidth=2,
               label=f'Mediana: {median_val:.6f}')
    
    # Crear caja de estadísticas con formato mejorado y unidades claras
    stats_text = (f'Estadísticas Detalladas\n'
                 f'━━━━━━━━━━━━━━━━━━━━\n'
                 f'Media:      {mean_val:.6f} u\n'  # 'u' representa unidades
                 f'Mediana:    {median_val:.6f} u\n'
                 f'Desv. Est.: {std_val:.6f} u\n'
                 f'Mínimo:     {np.min(mae_values):.6f} u\n'
                 f'Máximo:     {np.max(mae_values):.6f} u')
    
    # Añadir caja de texto con estadísticas en una posición optimizada
    bbox_props = dict(boxstyle="round,pad=0.5", fc=colors['text_box'], 
                     ec="gray", alpha=0.9)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=bbox_props,
            family='monospace')  # Usar fuente monospace para alineación
    
    # Configurar títulos y etiquetas con unidades claras
    ax.set_title(f'Distribución de MAE - {model_name}',
                fontsize=14, pad=20)
    ax.set_xlabel('Mean Absolute Error (MAE) [unidades]',
                 fontsize=12, labelpad=10)
    ax.set_ylabel('Densidad',
                 fontsize=12, labelpad=10)
    
    # Personalizar los ticks para mejor legibilidad
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Añadir leyenda en una posición optimizada
    ax.legend(fontsize=10, 
             framealpha=0.9,
             loc='upper right',
             bbox_to_anchor=(1.0, 0.95),
             title='Elementos del Gráfico')
    
    # Ajustar los límites del eje x basados en el rango intercuartílico
    q1, q3 = np.percentile(mae_values, [25, 75])
    iqr = q3 - q1
    ax.set_xlim(max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)
    
    # Ajustar el diseño
    plt.tight_layout()
    
    # Guardar la figura en alta resolución
    plt.savefig(save_dir / f'{model_name}_mae_distribution_{timestamp}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def calculate_batch_mae(model, data_loader, device, save_dir, model_name):
    """
    Calcula el MAE para cada imagen en el conjunto de datos y guarda los resultados.
    El MAE es útil porque proporciona una medida de error en las mismas unidades que los datos originales,
    lo que facilita su interpretación.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Listas para almacenar resultados
    mae_values = []
    sample_indices = []
    
    model.eval()
    print(f"\nCalculando MAE para el modelo: {model_name}")
    
    with torch.no_grad():
        for batch_idx, (signal, target) in enumerate(tqdm(data_loader)):
            signal, target = signal.to(device), target.to(device)
            
            # Generar predicción
            prediction = model(signal)
            
            # Calcular MAE para cada imagen en el batch
            for i in range(signal.size(0)):
                target_np = target[i, 0].cpu().numpy()
                pred_np = prediction[i, 0].cpu().numpy()
                
                # Calcular MAE
                mae = mean_absolute_error(target_np, pred_np)
                
                # Guardar resultados
                mae_values.append(mae)
                sample_indices.append(batch_idx * data_loader.batch_size + i)
                
                # Guardar visualizaciones de ejemplo
                if len(mae_values) <= 5:
                    save_sample_visualization(
                        signal[i, 0].cpu().numpy(),
                        target_np,
                        pred_np,
                        mae,
                        len(mae_values),
                        save_dir,
                        model_name
                    )
    
    # Convertir a array numpy para análisis
    mae_values = np.array(mae_values)
    
    # Crear DataFrame con los resultados
    results_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'mae': mae_values
    })
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar valores en CSV para análisis posterior
    results_df.to_csv(save_dir / f'{model_name}_mae_values_{timestamp}.csv', index=False)
    
    # Crear y guardar histograma
    create_enhanced_mae_histogram(mae_values, model_name, save_dir, timestamp)
    
    # Calcular y guardar estadísticas básicas
    stats = {
        'mean': np.mean(mae_values),
        'median': np.median(mae_values),
        'std': np.std(mae_values),
        'min': np.min(mae_values),
        'max': np.max(mae_values)
    }
    
    # Guardar estadísticas en formato legible
    with open(save_dir / f'{model_name}_mae_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Estadísticas MAE para {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Media: {stats['mean']:.6f}\n")
        f.write(f"Mediana: {stats['median']:.6f}\n")
        f.write(f"Desviación Estándar: {stats['std']:.6f}\n")
        f.write(f"Mínimo: {stats['min']:.6f}\n")
        f.write(f"Máximo: {stats['max']:.6f}\n")
    
    return mae_values

def save_sample_visualization(signal, target, prediction, mae, index, save_dir, model_name):
    """
    Guarda visualización de una muestra individual con su MAE.
    Incluye la señal original, la imagen objetivo y la predicción para una comparación completa.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Visualizar señal original
    im0 = axes[0].imshow(signal, cmap='viridis')
    axes[0].set_title('Input Signal')
    plt.colorbar(im0, ax=axes[0])
    
    # Visualizar imagen objetivo
    im1 = axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Target Image')
    plt.colorbar(im1, ax=axes[1])
    
    # Visualizar imagen reconstruida con MAE
    im2 = axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(f'Reconstructed Image\nMAE: {mae:.6f}')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_sample_{index}_mae_{mae:.6f}.png')
    plt.close()