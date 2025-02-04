import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

def load_metric_data(file_path, metric_name):
    """
    Carga los datos de las métricas desde un CSV con estructura específica.
    
    Args:
        file_path: Ruta al archivo CSV
        metric_name: Nombre de la métrica (mse, mae, etc.)
    
    Returns:
        np.array: Valores de la métrica
    """
    df = pd.read_csv(file_path)
    return df[metric_name.lower()].values

def calculate_confidence_intervals(pretrain_values, finetuned_values, metric_name, 
                                confidence_level=0.95, n_bootstrap=10000):
    """
    Calcula múltiples tipos de intervalos de confianza para una comprensión más profunda.
    
    Args:
        pretrain_values: Valores del modelo pre-entrenado
        finetuned_values: Valores del modelo fine-tuned
        metric_name: Nombre de la métrica
        confidence_level: Nivel de confianza (default: 0.95)
        n_bootstrap: Número de muestras bootstrap
    
    Returns:
        dict: Diferentes intervalos de confianza calculados
    """
    # 1. Intervalos de confianza paramétricos (t-distribution)
    def parametric_ci(data):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
        return {'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]}
    
    # 2. Intervalos de confianza bootstrap
    def bootstrap_ci(data):
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        ci = np.percentile(bootstrap_means, [(1-confidence_level)*100/2, 
                                           (1+confidence_level)*100/2])
        return {'mean': np.mean(data), 'ci_lower': ci[0], 'ci_upper': ci[1]}
    
    # 3. Intervalo de confianza para la diferencia
    differences = finetuned_values - pretrain_values
    diff_ci = bootstrap_ci(differences)
    
    # 4. Intervalos de confianza para percentiles
    def percentile_ci(data, percentile):
        bootstrap_percentiles = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_percentiles.append(np.percentile(bootstrap_sample, percentile))
        ci = np.percentile(bootstrap_percentiles, [(1-confidence_level)*100/2, 
                                                 (1+confidence_level)*100/2])
        return {'percentile': np.percentile(data, percentile), 
                'ci_lower': ci[0], 'ci_upper': ci[1]}
    
    return {
        'pretrain': {
            'parametric': parametric_ci(pretrain_values),
            'bootstrap': bootstrap_ci(pretrain_values),
            'percentiles': {
                'p25': percentile_ci(pretrain_values, 25),
                'p50': percentile_ci(pretrain_values, 50),
                'p75': percentile_ci(pretrain_values, 75)
            }
        },
        'finetuned': {
            'parametric': parametric_ci(finetuned_values),
            'bootstrap': bootstrap_ci(finetuned_values),
            'percentiles': {
                'p25': percentile_ci(finetuned_values, 25),
                'p50': percentile_ci(finetuned_values, 50),
                'p75': percentile_ci(finetuned_values, 75)
            }
        },
        'difference': diff_ci
    }

def visualize_confidence_intervals(ci_results, metric_name, save_dir):
    """
    Crea visualizaciones detalladas de los intervalos de confianza.
    
    Args:
        ci_results: Resultados de los intervalos de confianza
        metric_name: Nombre de la métrica
        save_dir: Directorio para guardar resultados
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Gráfico de Forest Plot para comparación de medias
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Pre-entrenado', 'Fine-tuned']
    means = [ci_results['pretrain']['parametric']['mean'],
             ci_results['finetuned']['parametric']['mean']]
    ci_lower = [ci_results['pretrain']['parametric']['ci_lower'],
                ci_results['finetuned']['parametric']['ci_lower']]
    ci_upper = [ci_results['pretrain']['parametric']['ci_upper'],
                ci_results['finetuned']['parametric']['ci_upper']]
    
    y_pos = np.arange(len(models))
    ax.errorbar(means, y_pos, xerr=[np.array(means)-np.array(ci_lower), 
                                   np.array(ci_upper)-np.array(means)],
                fmt='o', capsize=5, capthick=2, elinewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel(f'{metric_name} (con intervalos de confianza del 95%)')
    ax.set_title('Comparación de Medias con Intervalos de Confianza')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'ci_forest_plot_{metric_name}_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de la diferencia
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diff_mean = ci_results['difference']['mean']
    diff_ci = [ci_results['difference']['ci_lower'], 
               ci_results['difference']['ci_upper']]
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.errorbar([diff_mean], [0.5], xerr=[[diff_mean - diff_ci[0]], 
                [diff_ci[1] - diff_mean]],
                fmt='o', capsize=5, capthick=2, elinewidth=2)
    
    ax.set_yticks([])
    ax.set_xlabel(f'Diferencia en {metric_name} (Fine-tuned - Pre-entrenado)')
    ax.set_title('Intervalo de Confianza para la Diferencia')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'ci_difference_{metric_name}_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_confidence_results(ci_results, metric_name, save_dir):
    """
    Guarda los resultados detallados de los intervalos de confianza en un archivo.
    
    Args:
        ci_results: Resultados de los intervalos de confianza
        metric_name: Nombre de la métrica
        save_dir: Directorio para guardar resultados
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = save_dir / f'confidence_intervals_{metric_name}_{timestamp}.txt'
    
    with open(results_path, 'w') as f:
        f.write(f"Análisis Detallado de Intervalos de Confianza - {metric_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Resultados para cada modelo
        for model_name, results in [('Pre-entrenado', ci_results['pretrain']), 
                                  ('Fine-tuned', ci_results['finetuned'])]:
            f.write(f"{model_name}:\n")
            f.write("-" * 30 + "\n")
            
            # Intervalo paramétrico
            param = results['parametric']
            f.write("\nIntervalo Paramétrico (t-distribution):\n")
            f.write(f"Media: {param['mean']:.6f}\n")
            f.write(f"IC 95%: [{param['ci_lower']:.6f}, {param['ci_upper']:.6f}]\n")
            
            # Intervalo bootstrap
            boot = results['bootstrap']
            f.write("\nIntervalo Bootstrap:\n")
            f.write(f"Media: {boot['mean']:.6f}\n")
            f.write(f"IC 95%: [{boot['ci_lower']:.6f}, {boot['ci_upper']:.6f}]\n")
            
            # Intervalos para percentiles
            f.write("\nIntervalos para Percentiles:\n")
            for p_name, p_results in results['percentiles'].items():
                f.write(f"\n{p_name}:\n")
                f.write(f"Valor: {p_results['percentile']:.6f}\n")
                f.write(f"IC 95%: [{p_results['ci_lower']:.6f}, "
                       f"{p_results['ci_upper']:.6f}]\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
        
        # Resultados para la diferencia
        f.write("Diferencia (Fine-tuned - Pre-entrenado):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Media de la diferencia: {ci_results['difference']['mean']:.6f}\n")
        f.write(f"IC 95%: [{ci_results['difference']['ci_lower']:.6f}, "
               f"{ci_results['difference']['ci_upper']:.6f}]\n")
        
        # Interpretación
        f.write("\nInterpretación:\n")
        f.write("-" * 20 + "\n")
        
        diff_ci = [ci_results['difference']['ci_lower'], 
                  ci_results['difference']['ci_upper']]
        if 0 < diff_ci[0]:
            f.write("La mejora es estadísticamente significativa (IC no incluye 0)\n")
        elif diff_ci[1] < 0:
            f.write("Hay un deterioro estadísticamente significativo "
                   "(IC no incluye 0)\n")
        else:
            f.write("No hay evidencia concluyente de mejora o deterioro "
                   "(IC incluye 0)\n")

def main():
    """
    Función principal para calcular y visualizar intervalos de confianza detallados.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Calcular intervalos de confianza detallados para métricas")
    parser.add_argument('--pretrain_path', type=str, required=True,
                       help="Ruta al CSV con resultados del modelo pre-entrenado")
    parser.add_argument('--finetuned_path', type=str, required=True,
                       help="Ruta al CSV con resultados del modelo fine-tuned")
    parser.add_argument('--metric_name', type=str, required=True,
                       help="Nombre de la métrica (mse, mae, etc.)")
    parser.add_argument('--save_dir', type=str, default='confidence_results',
                       help="Directorio para guardar resultados")
    parser.add_argument('--confidence_level', type=float, default=0.95,
                       help="Nivel de confianza (default: 0.95)")
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                       help="Número de muestras bootstrap (default: 10000)")
    
    args = parser.parse_args()
    
    print(f"\nIniciando análisis de intervalos de confianza para {args.metric_name}")
    
    try:
        # Crear directorio de resultados
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar datos
        print("Cargando datos...")
        pretrain_values = load_metric_data(args.pretrain_path, args.metric_name)
        finetuned_values = load_metric_data(args.finetuned_path, args.metric_name)
        
        # Calcular intervalos de confianza
        print("Calculando intervalos de confianza...")
        ci_results = calculate_confidence_intervals(
            pretrain_values,
            finetuned_values,
            args.metric_name,
            args.confidence_level,
            args.n_bootstrap
        )
        
        # Generar visualizaciones
        print("Generando visualizaciones...")
        visualize_confidence_intervals(ci_results, args.metric_name, save_dir)
        
        # Guardar resultados
        print("Guardando resultados detallados...")
        save_confidence_results(ci_results, args.metric_name, save_dir)
        
        print(f"\nAnálisis de intervalos de confianza completado para {args.metric_name}")
        print(f"Resultados guardados en: {args.save_dir}")
        
        # Mostrar resumen rápido
        diff = ci_results['difference']
        print("\nResumen rápido:")
        print(f"Diferencia media: {diff['mean']:.6f}")
        print(f"IC 95%: [{diff['ci_lower']:.6f}, {diff['ci_upper']:.6f}]")
        
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")
        raise

if __name__ == "__main__":
    main()