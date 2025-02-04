import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from tqdm import tqdm

def load_metric_data(file_path, metric_name):
    """
    Carga y valida los datos de las métricas desde un CSV.
    
    Args:
        file_path: Ruta al archivo CSV con los resultados
        metric_name: Nombre de la métrica (mse, mae, etc.)
    
    Returns:
        np.array: Array con los valores de la métrica
    """
    try:
        # Cargar el CSV y verificar su estructura
        df = pd.read_csv(file_path)
        
        # Verificar que las columnas necesarias existen
        expected_columns = ['sample_idx', metric_name.lower()]
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"El CSV debe contener las columnas: {expected_columns}")
        
        # Verificar que no hay valores nulos
        if df[metric_name.lower()].isnull().any():
            raise ValueError(f"Se encontraron valores nulos en la columna {metric_name}")
        
        # Verificar que los índices son consecutivos
        if not all(df['sample_idx'] == np.arange(len(df))):
            print("Advertencia: Los índices de muestra no son consecutivos")
        
        # Extraer y retornar los valores de la métrica
        return df[metric_name.lower()].values
    
    except Exception as e:
        print(f"Error al cargar el archivo {file_path}: {str(e)}")
        raise

def perform_statistical_tests(pretrain_values, finetuned_values, metric_name, save_dir, alpha=0.05):
    """
    Realiza un análisis estadístico completo comparando los valores de las métricas
    entre el modelo pre-entrenado y el fine-tuned.
    
    Args:
        pretrain_values: Array con valores del modelo pre-entrenado
        finetuned_values: Array con valores del modelo fine-tuned
        metric_name: Nombre de la métrica (MSE, MAE, etc.)
        save_dir: Directorio para guardar resultados
        alpha: Nivel de significancia para las pruebas estadísticas
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcular estadísticas descriptivas
    stats_dict = {
        'Pre-entrenado': {
            'Media': np.mean(pretrain_values),
            'Mediana': np.median(pretrain_values),
            'Desv. Est.': np.std(pretrain_values),
            'Min': np.min(pretrain_values),
            'Max': np.max(pretrain_values)
        },
        'Fine-tuned': {
            'Media': np.mean(finetuned_values),
            'Mediana': np.median(finetuned_values),
            'Desv. Est.': np.std(finetuned_values),
            'Min': np.min(finetuned_values),
            'Max': np.max(finetuned_values)
        }
    }
    
    # Pruebas de normalidad
    shapiro_pretrain = stats.shapiro(pretrain_values)
    shapiro_finetuned = stats.shapiro(finetuned_values)
    
    # Pruebas de significancia
    t_stat, t_pval = stats.ttest_rel(pretrain_values, finetuned_values)
    w_stat, w_pval = stats.wilcoxon(pretrain_values, finetuned_values)
    
    # Calcular tamaño del efecto
    cohens_d = (np.mean(pretrain_values) - np.mean(finetuned_values)) / \
               np.sqrt((np.std(pretrain_values)**2 + np.std(finetuned_values)**2) / 2)
    
    # Calcular porcentaje de mejora
    improvement = ((np.mean(pretrain_values) - np.mean(finetuned_values)) / 
                  np.mean(pretrain_values)) * 100
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = save_dir / f'statistical_analysis_{metric_name}_{timestamp}.txt'
    
    with open(results_path, 'w') as f:
        # Encabezado
        f.write(f"Análisis Estadístico Detallado - {metric_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Estadísticas descriptivas
        f.write("Estadísticas Descriptivas\n")
        f.write("-" * 30 + "\n")
        for model, stats in stats_dict.items():
            f.write(f"\n{model}:\n")
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value:.6f}\n")
        
        # Mejora porcentual
        f.write(f"\nMejora Porcentual: {improvement:.2f}%\n\n")
        
        # Pruebas de normalidad
        f.write("Pruebas de Normalidad (Shapiro-Wilk)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Pre-entrenado: W={shapiro_pretrain[0]:.4f}, p={shapiro_pretrain[1]:.4e}\n")
        f.write(f"Fine-tuned: W={shapiro_finetuned[0]:.4f}, p={shapiro_finetuned[1]:.4e}\n\n")
        
        # Pruebas de significancia
        f.write("Pruebas de Significancia\n")
        f.write("-" * 30 + "\n")
        f.write(f"Prueba t pareada:\n")
        f.write(f"  Estadístico t = {t_stat:.4f}\n")
        f.write(f"  Valor p = {t_pval:.4e}\n\n")
        
        f.write(f"Prueba de Wilcoxon:\n")
        f.write(f"  Estadístico W = {w_stat:.4f}\n")
        f.write(f"  Valor p = {w_pval:.4e}\n\n")
        
        f.write(f"Tamaño del efecto (Cohen's d): {cohens_d:.4f}\n\n")
        
        # Interpretación
        f.write("Interpretación de Resultados\n")
        f.write("-" * 30 + "\n")
        
        # Interpretar normalidad
        f.write("Normalidad: ")
        if shapiro_pretrain[1] > alpha and shapiro_finetuned[1] > alpha:
            f.write("Ambas distribuciones son normales\n")
        else:
            f.write("Al menos una distribución no es normal\n")
        
        # Interpretar significancia
        f.write("Significancia estadística: ")
        if min(t_pval, w_pval) < alpha:
            f.write(f"Existe diferencia significativa (p < {alpha})\n")
        else:
            f.write(f"No hay diferencia significativa (p > {alpha})\n")
        
        # Interpretar tamaño del efecto
        f.write("Tamaño del efecto: ")
        if abs(cohens_d) < 0.2:
            f.write("Efecto pequeño\n")
        elif abs(cohens_d) < 0.5:
            f.write("Efecto mediano\n")
        else:
            f.write("Efecto grande\n")
    
    return {
        'stats': stats_dict,
        'improvement': improvement,
        'shapiro': (shapiro_pretrain, shapiro_finetuned),
        't_test': (t_stat, t_pval),
        'wilcoxon': (w_stat, w_pval),
        'cohens_d': cohens_d
    }

def create_comparison_visualization(pretrain_values, finetuned_values, metric_name, save_dir):
    """
    Crea visualizaciones detalladas comparando los modelos pre-entrenado y fine-tuned.
    
    Args:
        pretrain_values: Array con valores del modelo pre-entrenado
        finetuned_values: Array con valores del modelo fine-tuned
        metric_name: Nombre de la métrica
        save_dir: Directorio para guardar resultados
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 2)
    
    # 1. Boxplot comparativo
    ax1 = fig.add_subplot(gs[0, 0])
    data = [pretrain_values, finetuned_values]
    bp = ax1.boxplot(data, labels=['Pre-entrenado', 'Fine-tuned'])
    ax1.set_title(f'Comparación de {metric_name} - Boxplot')
    ax1.set_ylabel(metric_name)
    
    # 2. Violin plot
    ax2 = fig.add_subplot(gs[0, 1])
    df_violin = pd.DataFrame({
        'Modelo': ['Pre-entrenado']*len(pretrain_values) + ['Fine-tuned']*len(finetuned_values),
        metric_name: np.concatenate([pretrain_values, finetuned_values])
    })
    sns.violinplot(data=df_violin, x='Modelo', y=metric_name, ax=ax2)
    ax2.set_title(f'Distribución de {metric_name} - Violin Plot')
    
    # 3. Histograma superpuesto
    ax3 = fig.add_subplot(gs[1, :])
    sns.histplot(data=pretrain_values, label='Pre-entrenado', alpha=0.5, ax=ax3)
    sns.histplot(data=finetuned_values, label='Fine-tuned', alpha=0.5, ax=ax3)
    ax3.set_title(f'Distribución de {metric_name} - Histograma')
    ax3.set_xlabel(metric_name)
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / f'comparison_plots_{metric_name}_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Función principal para realizar pruebas estadísticas entre modelos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Realizar pruebas estadísticas entre modelos pre-entrenado y fine-tuned")
    parser.add_argument('--pretrain_path', type=str, required=True,
                       help="Ruta al CSV con resultados del modelo pre-entrenado")
    parser.add_argument('--finetuned_path', type=str, required=True,
                       help="Ruta al CSV con resultados del modelo fine-tuned")
    parser.add_argument('--metric_name', type=str, required=True,
                       help="Nombre de la métrica (mse, mae, etc.)")
    parser.add_argument('--save_dir', type=str, default='statistical_results',
                       help="Directorio para guardar resultados")
    args = parser.parse_args()
    
    print(f"\nIniciando análisis estadístico para {args.metric_name}")
    
    try:
        # Cargar datos
        print("Cargando datos...")
        pretrain_values = load_metric_data(args.pretrain_path, args.metric_name)
        finetuned_values = load_metric_data(args.finetuned_path, args.metric_name)
        
        # Verificar que tenemos el mismo número de muestras
        if len(pretrain_values) != len(finetuned_values):
            raise ValueError("Los conjuntos de datos tienen diferentes tamaños")
        
        print(f"Datos cargados exitosamente: {len(pretrain_values)} muestras por modelo")
        
        # Realizar pruebas estadísticas
        print("Realizando pruebas estadísticas...")
        results = perform_statistical_tests(
            pretrain_values,
            finetuned_values,
            args.metric_name,
            args.save_dir
        )
        
        # Crear visualizaciones
        print("Generando visualizaciones...")
        create_comparison_visualization(
            pretrain_values,
            finetuned_values,
            args.metric_name,
            Path(args.save_dir)
        )
        
        print(f"\nAnálisis estadístico completado para {args.metric_name}")
        print(f"Resultados guardados en: {args.save_dir}")
        
        # Mostrar resumen rápido
        print("\nResumen rápido:")
        print(f"Mejora porcentual: {results['improvement']:.2f}%")
        print(f"Significancia estadística: {'Sí' if min(results['t_test'][1], results['wilcoxon'][1]) < 0.05 else 'No'}")
        
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()