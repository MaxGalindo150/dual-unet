import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def calculate_psnr(target, prediction, data_range=None):
    """
    Calcula el Peak Signal-to-Noise Ratio entre la imagen objetivo y la predicha.
    
    Args:
        target: Imagen objetivo (numpy array)
        prediction: Imagen predicha (numpy array)
        data_range: Rango dinámico de las imágenes (max-min). Si es None, se calcula del target.
    
    Returns:
        float: Valor PSNR en dB
    """
    if data_range is None:
        data_range = target.max() - target.min()
    
    mse = np.mean((target - prediction) ** 2)
    
    if mse == 0:  # Las imágenes son idénticas
        return float('inf')
    
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr

def create_enhanced_psnr_histogram(psnr_values, model_name, save_dir, timestamp):
    """
    Crea un histograma mejorado de la distribución de PSNR con elementos visuales adicionales.
    PSNR es una métrica en decibelios (dB) donde valores más altos indican mejor calidad.
    Típicamente:
    - > 40 dB: Excelente
    - > 35 dB: Bueno
    - > 30 dB: Aceptable
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    # Paleta de colores optimizada para PSNR
    colors = {
        'hist': '#3498db',      # Azul para el histograma
        'mean': '#e74c3c',      # Rojo para la media
        'median': '#2ecc71',    # Verde para la mediana
        'kde': '#2c3e50'        # Azul oscuro para KDE
    }
    
    # Definir colores específicos para cada umbral de calidad
    threshold_colors = {
        40: '#FF0000',  # Rojo para excelente
        35: '#FFA500',  # Naranja para bueno
        30: '#FFD700'   # Dorado para aceptable
    }
    
    # Crear el histograma principal
    sns.histplot(data=psnr_values, 
                bins=30,
                color=colors['hist'],
                alpha=0.7,
                stat='density',
                ax=ax,
                label='Histograma')
    
    # Añadir la curva de densidad KDE
    sns.kdeplot(data=psnr_values,
                color=colors['kde'],
                linewidth=2,
                ax=ax,
                label='Densidad KDE')
    
    # Calcular estadísticas
    mean_val = np.mean(psnr_values)
    median_val = np.median(psnr_values)
    std_val = np.std(psnr_values)
    
    # Añadir líneas verticales para media y mediana
    ax.axvline(mean_val, color=colors['mean'], linestyle='--', linewidth=2,
               label=f'Media: {mean_val:.2f} dB')
    ax.axvline(median_val, color=colors['median'], linestyle='--', linewidth=2,
               label=f'Mediana: {median_val:.2f} dB')
    
    # Definir umbrales de calidad
    quality_thresholds = {
        40: 'Excelente',
        35: 'Bueno',
        30: 'Aceptable'
    }
    
    # Dibujar las líneas de umbral con mejor visibilidad
    for threshold, label in quality_thresholds.items():
        if threshold > min(psnr_values) and threshold < max(psnr_values):
            # Dibujar la línea vertical
            ax.axvline(threshold, 
                      color=threshold_colors[threshold],
                      linestyle='--',
                      linewidth=2,
                      alpha=0.8,
                      label=f'{label} ({threshold} dB)')
            
            # Añadir texto directamente sobre la línea
            ax.text(threshold, 
                   ax.get_ylim()[1] * 0.95,
                   f'{label}\n({threshold} dB)',
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
                 f'Media:      {mean_val:.2f} dB\n'
                 f'Mediana:    {median_val:.2f} dB\n'
                 f'Desv. Est.: {std_val:.2f} dB\n'
                 f'Mínimo:     {np.min(psnr_values):.2f} dB\n'
                 f'Máximo:     {np.max(psnr_values):.2f} dB\n\n'
                 f'Interpretación:\n'
                 f'━━━━━━━━━━━━━\n'
                 f'> 40 dB: Excelente\n'
                 f'> 35 dB: Bueno\n'
                 f'> 30 dB: Aceptable\n'
                 f'< 30 dB: Pobre')
    
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
    ax.set_title(f'Distribución de PSNR - {model_name}',
                fontsize=14, pad=20)
    ax.set_xlabel('Peak Signal-to-Noise Ratio (PSNR) [dB]',
                 fontsize=12, labelpad=10)
    ax.set_ylabel('Densidad',
                 fontsize=12, labelpad=10)
    
    # Ajustar límites del eje x al rango de datos
    data_min = np.min(psnr_values)
    data_max = np.max(psnr_values)
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
    plt.savefig(save_dir / f'{model_name}_psnr_distribution_{timestamp}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def calculate_batch_psnr(model, data_loader, device, save_dir, model_name):
    """
    Calcula el PSNR para cada imagen en el conjunto de datos y guarda los resultados.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    psnr_values = []
    sample_indices = []
    
    model.eval()
    print(f"\nCalculando PSNR para el modelo: {model_name}")
    
    with torch.no_grad():
        for batch_idx, (signal, target) in enumerate(tqdm(data_loader)):
            signal, target = signal.to(device), target.to(device)
            prediction = model(signal)
            
            for i in range(signal.size(0)):
                target_np = target[i, 0].cpu().numpy()
                pred_np = prediction[i, 0].cpu().numpy()
                
                # Calcular PSNR
                psnr = calculate_psnr(target_np, pred_np)
                
                psnr_values.append(psnr)
                sample_indices.append(batch_idx * data_loader.batch_size + i)
                
                # Guardar visualizaciones de ejemplo
                if len(psnr_values) <= 5:
                    save_sample_visualization(
                        signal[i, 0].cpu().numpy(),
                        target_np,
                        pred_np,
                        psnr,
                        len(psnr_values),
                        save_dir,
                        model_name
                    )
    
    psnr_values = np.array(psnr_values)
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'psnr': psnr_values,
        'quality_category': pd.cut(psnr_values, 
                                 bins=[-np.inf, 30, 35, 40, np.inf],
                                 labels=['Pobre', 'Aceptable', 'Bueno', 'Excelente'])
    })
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(save_dir / f'{model_name}_psnr_values_{timestamp}.csv', index=False)
    
    # Crear visualizaciones
    create_enhanced_psnr_histogram(psnr_values, model_name, save_dir, timestamp)
    
    # Calcular estadísticas de calidad
    quality_stats = results_df['quality_category'].value_counts().to_dict()
    total_samples = len(psnr_values)
    
    # Guardar estadísticas detalladas
    with open(save_dir / f'{model_name}_psnr_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Estadísticas PSNR para {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Media: {np.mean(psnr_values):.2f} dB\n")
        f.write(f"Mediana: {np.median(psnr_values):.2f} dB\n")
        f.write(f"Desviación Estándar: {np.std(psnr_values):.2f} dB\n")
        f.write(f"Mínimo: {np.min(psnr_values):.2f} dB\n")
        f.write(f"Máximo: {np.max(psnr_values):.2f} dB\n\n")
        
        f.write("Distribución de Calidad:\n")
        f.write("-" * 30 + "\n")
        for category, count in quality_stats.items():
            percentage = (count / total_samples) * 100
            f.write(f"{category}: {count} imágenes ({percentage:.1f}%)\n")
    
    return psnr_values

def save_sample_visualization(signal, target, prediction, psnr, index, save_dir, model_name):
    """
    Guarda visualización de una muestra individual con su valor PSNR y mapa de diferencias.
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
    axes[2].set_title(f'Reconstructed Image\nPSNR: {psnr:.2f} dB')
    plt.colorbar(im2, ax=axes[2])
    
    # Mapa de diferencias
    difference = np.abs(target - prediction)
    im3 = axes[3].imshow(difference, cmap='hot')
    axes[3].set_title('Difference Map')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_sample_{index}_psnr_{psnr:.2f}.png')
    plt.close()