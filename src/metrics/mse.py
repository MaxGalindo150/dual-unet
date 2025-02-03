import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def create_enhanced_mse_histogram(mse_values, model_name, save_dir, timestamp):
    """
    Crea un histograma mejorado de la distribución de MSE con elementos visuales adicionales.
    La función asegura una disposición clara de todos los elementos sin superposición.
    """
    # Configurar el estilo general
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear una figura más ancha para dar más espacio
    fig = plt.figure(figsize=(14, 8))
    
    # Crear el grid para el histograma y la caja de estadísticas
    gs = plt.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    # Colores personalizados con mejor contraste
    colors = {
        'hist': '#3498db',      # Azul para el histograma
        'mean': '#e74c3c',      # Rojo para la media
        'median': '#2ecc71',    # Verde para la mediana
        'kde': '#2c3e50',       # Azul oscuro para KDE
        'text_box': '#f8f9fa'   # Gris claro para el fondo del texto
    }
    
    # Crear el histograma principal
    sns.histplot(data=mse_values, 
                bins=30,
                color=colors['hist'],
                alpha=0.7,
                stat='density',
                ax=ax,
                label='Histograma')
    
    # Añadir la curva de densidad KDE
    sns.kdeplot(data=mse_values,
                color=colors['kde'],
                linewidth=2,
                ax=ax,
                label='Densidad KDE')
    
    # Calcular estadísticas
    mean_val = np.mean(mse_values)
    median_val = np.median(mse_values)
    std_val = np.std(mse_values)
    
    # Añadir líneas verticales para media y mediana
    ax.axvline(mean_val, color=colors['mean'], linestyle='--', linewidth=2,
               label=f'Media: {mean_val:.6f}')
    ax.axvline(median_val, color=colors['median'], linestyle='--', linewidth=2,
               label=f'Mediana: {median_val:.6f}')
    
    # Crear caja de estadísticas con formato mejorado
    stats_text = (f'Estadísticas Detalladas\n'
                 f'━━━━━━━━━━━━━━━━━━━━\n'
                 f'Media:      {mean_val:.6f}\n'
                 f'Mediana:    {median_val:.6f}\n'
                 f'Desv. Est.: {std_val:.6f}\n'
                 f'Mínimo:     {np.min(mse_values):.6f}\n'
                 f'Máximo:     {np.max(mse_values):.6f}')
    
    # Añadir caja de texto con estadísticas en una posición que no se superponga
    bbox_props = dict(boxstyle="round,pad=0.5", fc=colors['text_box'], 
                     ec="gray", alpha=0.9)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=bbox_props,
            family='monospace')  # Usar fuente monospace para alineación
    
    # Configurar títulos y etiquetas
    ax.set_title(f'Distribución de MSE - {model_name}',
                fontsize=14, pad=20)
    ax.set_xlabel('Mean Squared Error (MSE)',
                 fontsize=12, labelpad=10)
    ax.set_ylabel('Densidad',
                 fontsize=12, labelpad=10)
    
    # Personalizar los ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Añadir leyenda en una mejor posición
    ax.legend(fontsize=10, 
             framealpha=0.9,
             loc='upper right',
             bbox_to_anchor=(1.0, 0.95),
             title='Elementos del Gráfico')
    
    # Ajustar los límites del eje x para mejor visualización
    q1, q3 = np.percentile(mse_values, [25, 75])
    iqr = q3 - q1
    ax.set_xlim(max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)
    
    # Ajustar el diseño con más espacio para la leyenda
    plt.tight_layout()
    
    # Guardar la figura con alta resolución
    plt.savefig(save_dir / f'{model_name}_mse_distribution_{timestamp}.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()


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
    create_enhanced_mse_histogram(mse_values, model_name, save_dir, timestamp)
    
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

