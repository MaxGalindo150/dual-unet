from scipy import stats
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def run_statistical_tests(pretrain_values, finetune_values, save_dir):
    metrics = {}
    for metric in ['MSE', 'MAE', 'SSIM']:
        t_stat, p_value = stats.ttest_rel(pretrain_values[metric], finetune_values[metric])
        w_stat, w_p_value = stats.wilcoxon(pretrain_values[metric], finetune_values[metric])
        cohens_d = (np.mean(finetune_values[metric]) - np.mean(pretrain_values[metric])) / np.sqrt(
            (np.var(finetune_values[metric]) + np.var(pretrain_values[metric])) / 2)
        
        metrics[metric] = {
            't_test': {'statistic': float(t_stat), 'p_value': float(p_value)},
            'wilcoxon': {'statistic': float(w_stat), 'p_value': float(w_p_value)},
            'cohens_d': float(cohens_d)
        }
    
    save_results(metrics, save_dir)
    visualize_results(metrics, save_dir)
    
def save_results(metrics, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{save_dir}/significance_tests.txt', 'w') as f:
        f.write("Pruebas Estadísticas de Significancia\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, results in metrics.items():
            f.write(f"\nMétrica: {metric}\n")
            f.write("-" * 20 + "\n")
            
            f.write("\nT-test Pareado:\n")
            f.write(f"Estadístico: {results['t_test']['statistic']:.6f}\n")
            f.write(f"p-valor: {results['t_test']['p_value']:.6f}\n")
            f.write(f"Significativo: {'Sí' if results['t_test']['p_value'] < 0.05 else 'No'}\n")
            
            f.write("\nWilcoxon Signed-Rank Test:\n")
            f.write(f"Estadístico: {results['wilcoxon']['statistic']:.6f}\n")
            f.write(f"p-valor: {results['wilcoxon']['p_value']:.6f}\n")
            f.write(f"Significativo: {'Sí' if results['wilcoxon']['p_value'] < 0.05 else 'No'}\n")
            
            f.write("\nTamaño del Efecto (Cohen's d):\n")
            f.write(f"d = {results['cohens_d']:.6f}\n")
            f.write(f"Interpretación: {interpret_cohens_d(results['cohens_d'])}\n")

def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2: return "Efecto insignificante"
    elif d < 0.5: return "Efecto pequeño"
    elif d < 0.8: return "Efecto mediano"
    else: return "Efecto grande"

def visualize_results(metrics, save_dir):
    plt.figure(figsize=(12, 6))
    
    metrics_df = []
    for metric, results in metrics.items():
        metrics_df.append({
            'Metric': metric,
            'Cohen\'s d': abs(results['cohens_d']),
            'Significant': results['t_test']['p_value'] < 0.05
        })
    
    colors = ['green' if m['Significant'] else 'red' for m in metrics_df]
    plt.bar([m['Metric'] for m in metrics_df], 
           [m['Cohen\'s d'] for m in metrics_df],
           color=colors)
    
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.3, label='Efecto pequeño')
    plt.axhline(y=0.5, color='y', linestyle='--', alpha=0.3, label='Efecto mediano')
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='Efecto grande')
    
    plt.title('Tamaño del Efecto por Métrica')
    plt.ylabel('Cohen\'s d (valor absoluto)')
    plt.legend()
    plt.savefig(f'{save_dir}/effect_sizes.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Análisis estadístico de métricas de reconstrucción.")
    parser.add_argument('--pretrain_metrics', type=str, required=True, 
                      help="Archivo JSON con métricas de preentrenamiento")
    parser.add_argument('--finetune_metrics', type=str, required=True, 
                      help="Archivo JSON con métricas de fine-tuning")
    parser.add_argument('--save_dir', type=str, required=True, 
                      help="Directorio para guardar resultados")
    
    args = parser.parse_args()
    
    pretrain_metrics = load_metrics(args.pretrain_metrics)
    finetune_metrics = load_metrics(args.finetune_metrics)
    
    run_statistical_tests(pretrain_metrics, finetune_metrics, args.save_dir)

if __name__ == "__main__":
    main()
    
    