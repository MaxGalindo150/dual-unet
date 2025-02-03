import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def create_enhanced_ssim_histogram(ssim_values, model_name, save_dir, timestamp):
    """
    Crea un histograma mejorado de la distribución de SSIM con elementos visuales adicionales.
    El SSIM es una métrica que evalúa la similitud estructural entre imágenes, con valores
    entre -1 y 1, donde:
    - 1.0 indica imágenes idénticas
    - 0.0 indica no hay similitud estructural
    - -1.0 indica imágenes completamente diferentes
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig = plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    # Paleta de colores optimizada para SSIM
    colors = {
        'hist': '#3498db',      # Azul para el histograma
        'mean': '#e74c3c',      # Rojo para la media
        'median': '#2ecc71',    # Verde para la mediana
        'kde': '#2c3e50',       # Azul oscuro para KDE
        'text_box': '#f8f9fa',  # Gris claro para el fondo del texto
        'threshold': '#f39c12'  # Naranja para umbrales de calidad
    }
    
    # Crear el histograma principal
    sns.histplot(data=ssim_values, 
                bins=30,
                color=colors['hist'],
                alpha=0.7,
                stat='density',
                ax=ax,
                label='Histograma')
    
    # Añadir la curva de densidad KDE
    sns.kdeplot(data=ssim_values,
                color=colors['kde'],
                linewidth=2,
                ax=ax,
                label='Densidad KDE')
    
    # Calcular estadísticas
    mean_val = np.mean(ssim_values)
    median_val = np.median(ssim_values)
    std_val = np.std(ssim_values)
    
    # Añadir líneas verticales para media y mediana
    ax.axvline(mean_val, color=colors['mean'], linestyle='--', linewidth=2,
               label=f'Media: {mean_val:.4f}')
    ax.axvline(median_val, color=colors['median'], linestyle='--', linewidth=2,
               label=f'Mediana: {median_val:.4f}')
    
    # Añadir líneas de umbral de calidad
    quality_thresholds = {
        0.97: 'Excelente',
        0.95: 'Bueno',
        0.90: 'Aceptable'
    }
    
    for threshold, label in quality_thresholds.items():
        if threshold > min(ssim_values) and threshold < max(ssim_values):
            ax.axvline(threshold, color=colors['threshold'], linestyle=':', linewidth=1,
                      alpha=0.5, label=f'{label} ({threshold:.2f})')
    
    # Crear caja de estadísticas con interpretación
    stats_text = (f'Estadísticas Detalladas\n'
                 f'━━━━━━━━━━━━━━━━━━━━\n'
                 f'Media:      {mean_val:.4f}\n'
                 f'Mediana:    {median_val:.4f}\n'
                 f'Desv. Est.: {std_val:.4f}\n'
                 f'Mínimo:     {np.min(ssim_values):.4f}\n'
                 f'Máximo:     {np.max(ssim_values):.4f}\n\n'
                 f'Interpretación:\n'
                 f'━━━━━━━━━━━━━\n'
                 f'> 0.90: Excelente\n'
                 f'> 0.80: Bueno\n'
                 f'> 0.70: Aceptable\n'
                 f'< 0.70: Pobre')
    
    # Añadir caja de texto con estadísticas
    bbox_props = dict(boxstyle="round,pad=0.5", fc=colors['text_box'], 
                     ec="gray", alpha=0.9)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=bbox_props,
            family='monospace')
    
    # Configurar títulos y etiquetas
    ax.set_title(f'Distribución de SSIM - {model_name}',
                fontsize=14, pad=20)
    ax.set_xlabel('Structural Similarity Index (SSIM)',
                 fontsize=12, labelpad=10)
    ax.set_ylabel('Densidad',
                 fontsize=12, labelpad=10)
    
    # Ajustar límites del eje x al rango válido de SSIM
    data_min = np.min(ssim_values)
    data_max = np.max(ssim_values)
    
    
    margin = (data_max - data_min) * 0.01
    x_min = max(0, data_min - margin)  
    x_max = min(1, data_max + margin)  
    
    ax.set_xlim(x_min, x_max)
    
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
    plt.savefig(save_dir / f'{model_name}_ssim_distribution_{timestamp}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def calculate_batch_ssim(model, data_loader, device, save_dir, model_name):
    """
    Calcula el SSIM para cada imagen en el conjunto de datos y guarda los resultados.
    El SSIM evalúa la similitud estructural entre las imágenes originales y reconstruidas,
    considerando luminancia, contraste y estructura.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ssim_values = []
    sample_indices = []
    
    model.eval()
    print(f"\nCalculando SSIM para el modelo: {model_name}")
    
    with torch.no_grad():
        for batch_idx, (signal, target) in enumerate(tqdm(data_loader)):
            signal, target = signal.to(device), target.to(device)
            prediction = model(signal)
            
            for i in range(signal.size(0)):
                target_np = target[i, 0].cpu().numpy()
                pred_np = prediction[i, 0].cpu().numpy()
                
                # Calcular SSIM
                ssim_val = ssim(target_np, pred_np, 
                              data_range=target_np.max() - target_np.min())
                
                ssim_values.append(ssim_val)
                sample_indices.append(batch_idx * data_loader.batch_size + i)
                
                # Guardar visualizaciones de ejemplo
                if len(ssim_values) <= 5:
                    save_sample_visualization(
                        signal[i, 0].cpu().numpy(),
                        target_np,
                        pred_np,
                        ssim_val,
                        len(ssim_values),
                        save_dir,
                        model_name
                    )
    
    ssim_values = np.array(ssim_values)
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'sample_idx': sample_indices,
        'ssim': ssim_values,
        'quality_category': pd.cut(ssim_values, 
                                 bins=[-1, 0.7, 0.8, 0.9, 1],
                                 labels=['Pobre', 'Aceptable', 'Bueno', 'Excelente'])
    })
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(save_dir / f'{model_name}_ssim_values_{timestamp}.csv', index=False)
    
    # Crear visualizaciones
    create_enhanced_ssim_histogram(ssim_values, model_name, save_dir, timestamp)
    
    # Calcular estadísticas de calidad
    quality_stats = results_df['quality_category'].value_counts().to_dict()
    total_samples = len(ssim_values)
    
    # Guardar estadísticas detalladas
    with open(save_dir / f'{model_name}_ssim_stats_{timestamp}.txt', 'w') as f:
        f.write(f"Estadísticas SSIM para {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Media: {np.mean(ssim_values):.4f}\n")
        f.write(f"Mediana: {np.median(ssim_values):.4f}\n")
        f.write(f"Desviación Estándar: {np.std(ssim_values):.4f}\n")
        f.write(f"Mínimo: {np.min(ssim_values):.4f}\n")
        f.write(f"Máximo: {np.max(ssim_values):.4f}\n\n")
        
        f.write("Distribución de Calidad:\n")
        f.write("-" * 30 + "\n")
        for category, count in quality_stats.items():
            percentage = (count / total_samples) * 100
            f.write(f"{category}: {count} imágenes ({percentage:.1f}%)\n")
    
    return ssim_values

def save_sample_visualization(signal, target, prediction, ssim_val, index, save_dir, model_name):
    """
    Guarda visualización de una muestra individual con su valor SSIM.
    Incluye un mapa de diferencias para visualizar dónde ocurren las principales
    diferencias estructurales.
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
    axes[2].set_title(f'Reconstructed Image\nSSIM: {ssim_val:.4f}')
    plt.colorbar(im2, ax=axes[2])
    
    # Mapa de diferencias
    difference = np.abs(target - prediction)
    im3 = axes[3].imshow(difference, cmap='hot')
    axes[3].set_title('Difference Map')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_sample_{index}_ssim_{ssim_val:.4f}.png')
    plt.close()