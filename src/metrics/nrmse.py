import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def calculate_nrmse(target, prediction, normalization='range'):
    """
    Calcula el Normalized Root Mean Square Error entre la imagen objetivo y la predicha.
    
    Args:
        target: Imagen objetivo (numpy array)
        prediction: Imagen predicha (numpy array)
        normalization: Método de normalización ('range' o 'mean')
    
    Returns:
        float: Valor NRMSE (sin unidades, al ser normalizado)
    """
    # Calcular MSE
    mse = np.mean((target - prediction) ** 2)
    rmse = np.sqrt(mse)
    
    if normalization == 'range':
        # Normalizar por el rango de valores
        normalization_factor = target.max() - target.min()
    else:  # normalization == 'mean'
        # Normalizar por la media
        normalization_factor = np.mean(np.abs(target))
    
    # Evitar división por cero
    if normalization_factor == 0:
        return float('inf')
    
    return rmse / normalization_factor

def create_enhanced_nrmse_histogram(nrmse_values, model_name, save_dir, timestamp):
    """
    Crea un histograma mejorado de la distribución de NRMSE con elementos visuales adicionales.
    NRMSE es una métrica normalizada donde valores más bajos indican mejor calidad.
    Típicamente:
    - < 0.05: Excelente
    - < 0.10: Bueno
    - < 0.15: Aceptable
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    # Paleta de colores optimizada
    colors = {
        'hist': '#3498db',      # Azul para el histograma
        'mean': '#e74c3c',      # Rojo para la media
        'median': '#2ecc71',    # Verde para la mediana
        'kde': '#2c3e50'        # Azul oscuro para KDE
    }
    
    # Definir colores específicos para cada umbral de calidad
    threshold_colors = {
        0.05: '#FF0000',  # Rojo para excelente
        0.10: '#FFA500',  # Naranja para bueno
        0.15: '#FFD700'   # Dorado para aceptable
    }
    
    # Crear el histograma principal
    sns.histplot(data=nrmse_values, 
                bins=30,
                color=colors['hist'],
                alpha=0.7,
                stat='density',
                ax=ax,
                label='Histograma')
    
    # Añadir la curva de densidad KDE
    sns.kdeplot(data=nrmse_values,
                color=colors['kde'],
                linewidth=2,
                ax=ax,
                label='Densidad KDE')
    
    # Calcular estadísticas
    mean_val = np.mean(nrmse_values)
    median_val = np.median(nrmse_values)
    std_val = np.std(nrmse_values)
    
    # Añadir líneas verticales para media y mediana
    ax.axvline(mean_val, color=colors['mean'], linestyle='--', linewidth=2,
               label=f'Media: {mean_val:.4f}')
    ax.axvline(median_val, color=colors['median'], linestyle='--', linewidth=2,
               label=f'Mediana: {median_val:.4f}')
    
    # Definir umbrales de calidad
    quality_thresholds = {
        0.05: 'Excelente',
        0.10: 'Bueno',
        0.15: 'Aceptable'
    }
    
    # Dibujar las líneas de umbral con mejor visibilidad
    for threshold, label in quality_thresholds.items():
        if threshold > min(nrmse_values) and threshold < max(nrmse_values):
            # Dibujar la línea vertical
            ax.axvline(threshold, 
                      color=threshold_colors[threshold],
                      linestyle='--',
                      linewidth=2,
                      alpha=0.8,
                      label=f'{label} ({threshold:.3f})')
            
            # Añadir texto directamente sobre la línea
            ax.text(threshold, 
                   ax.get_ylim()[1] * 0.95,
                   f'{label}\n({threshold:.3f})',
                   rotation=90,
                   verticalalignment='top',
                   horizontalalignment='right',
                   color=threshold_colors[threshold],
                   fontweight='bold',
                   bbox=dict(facecolor='white', 
                            alpha=0.8,
                            edgecolor='none',
                            pad=2))
    
    # Crear caja de estadísticas con interpretación
    stats_text = (f'Estadísticas Detalladas\n'
                 f'━━━━━━━━━━━━━━━━━━━━\n'
                 f'Media:      {mean_val:.4f}\n'
                 f'Mediana:    {median_val:.4f}\n'
                 f'Desv. Est.: {std_val:.4f}\n'
                 f'Mínimo:     {np.min(nrmse_values):.4f}\n'
                 f'Máximo:     {np.max(nrmse_values):.4f}\n\n'
                 f'Interpretación:\n'
                 f'━━━━━━━━━━━━━\n'
                 f'< 0.05: Excelente\n'
                 f'< 0.10: Bueno\n'
                 f'< 0.15: Aceptable\n'
                 f'> 0.15: Pobre')
    
    # Añadir caja de texto con estadísticas
    bbox_props = dict(boxstyle="round,pad=0.5", fc='white', 
                     ec="gray", alpha=0.9)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=bbox_props,
            family='monospace')
    
    # Configurar títulos y etiquetas
    ax.set_title(f'Distribución de NRMSE - {model_name}',
                fontsize=14, pad=20)
    ax.set_xlabel('Normalized Root Mean Square Error (NRMSE)',
                 fontsize=12, labelpad=10)
    ax.set_ylabel('Densidad',
                 fontsize=12, labelpad=10)
    
    # Ajustar límites del eje x al rango de datos
    data_min = np.min(nrmse_values)
    data_max = np.max(nrmse_values)
    margin = (data_max - data_min) * 0.05  # 5% de margen
    ax.set_xlim(data_min - margin, data_max + margin)
    
    # Ajustar el número de ticks para mejor legibilidad
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    
    # Personalizar los ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Añadir leyenda
    ax.legend(fontsize=10, 
             framealpha=0.9,
             loc='upper left',
             bbox_to_anchor=(1.0, 1.0),
             title='Elementos del Gráfico')
    
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig(save_dir / f'{model_name}_nrmse_distribution_{timestamp}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def calculate_batch_nrmse(model, data_loader, device, save_dir, model_name):
    """
    Calcula el NRMSE para cada imagen en el conjunto de datos y guarda los resultados.
    El NRMSE es especialmente útil para comparar errores entre diferentes datasets
    ya que está normalizado.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    nrmse_values = []
    sample_indices = []
    
    model.eval()
    print(f"\nCalculando NRMSE para el modelo: {model_name}")
    
    with torch.no_grad():
        for batch_idx, (signal, target) in enumerate(tqdm(data_loader)):
            signal, target = signal.to(device), target.to(device)
            prediction = model(signal)
            
            for i in range(signal.size(0)):
                target_np = target[i, 0].cpu().numpy()
                pred_np = prediction[i, 0].cpu().numpy()
                
                # Calcular NRMSE
                nrmse = calculate_nrmse(target_np, pred_np, normalization='range')
                
                nrmse_values.append(nrmse)
                sample_indices.append(batch_idx * data_loader.batch_size + i)
                
                # Guardar visualizaciones de ejemplo
                if len(nrmse_values) <= 5:
                    save_sample_visualization(
                        signal[i, 0].cpu().numpy(),
                        target_np,
                        pred_np,
                        nrmse,
                        len(nrmse_values),
                        save_dir,
                        model_name
                    )
    
    nrmse_values = np.array(nrmse_values)
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'nrmse': nrmse_values,
        'quality_category': pd.cut(nrmse_values, 
                                 bins=[-np.inf, 0.05, 0.10, 0.15, np.inf],
                                 labels=['Excelente', 'Bueno', 'Aceptable', 'Pobre'])
    })
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(save_dir / f'{model_name}_nrmse_values_{timestamp}.csv', index=False)
    
    # Crear visualizaciones
    create_enhanced_nrmse_histogram(nrmse_values, model_name, save_dir, timestamp)
    
    # Calcular estadísticas de calidad
    quality_stats = results_df['quality_category'].value_counts().to_dict()
    total_samples = len(nrmse_values)
    
    # Guardar estadísticas detalladas
    with open(save_dir / f'{model_name}_nrmse_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Estadísticas NRMSE para {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Media: {np.mean(nrmse_values):.4f}\n")
        f.write(f"Mediana: {np.median(nrmse_values):.4f}\n")
        f.write(f"Desviación Estándar: {np.std(nrmse_values):.4f}\n")
        f.write(f"Mínimo: {np.min(nrmse_values):.4f}\n")
        f.write(f"Máximo: {np.max(nrmse_values):.4f}\n\n")
        
        f.write("Distribución de Calidad:\n")
        f.write("-" * 30 + "\n")
        for category, count in quality_stats.items():
            percentage = (count / total_samples) * 100
            f.write(f"{category}: {count} imágenes ({percentage:.1f}%)\n")
    
    return nrmse_values

def save_sample_visualization(signal, target, prediction, nrmse, index, save_dir, model_name):
    """
    Guarda visualización de una muestra individual con su valor NRMSE y mapa de diferencias.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
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
    axes[2].set_title(f'Reconstructed Image\nNRMSE: {nrmse:.4f}')
    plt.colorbar(im2, ax=axes[2])
    
    # Mapa de diferencias normalizado
    difference = np.abs(target - prediction) / (target.max() - target.min())
    im3 = axes[3].imshow(difference, cmap='hot')
    axes[3].set_title('Normalized Difference Map')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_sample_{index}_nrmse_{nrmse:.4f}.png')
    plt.close()