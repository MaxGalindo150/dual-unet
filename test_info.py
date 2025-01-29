import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
from attention_unet_model import AttentionUNet
from physics_informed import PhysicsInformedWrapper, PATLoss
from preprocess.preprocess_simulated_data import load_and_preprocess_data
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from datetime import datetime

def evaluate_physics_metrics(model, test_loader, device):
    """
    Evalúa métricas específicas para el modelo físicamente informado
    """
    physics_metrics = {
        'signal_consistency': [],  # Consistencia entre señales simuladas y reales
        'uncertainty': [],        # Valores de incertidumbre
        'physics_loss': []        # Pérdida física
    }
    
    criterion = PATLoss(alpha=0.5)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluando métricas físicas"):
            data, target = data.to(device), target.to(device)
            
            if isinstance(model, PhysicsInformedWrapper):
                refined, simulated, uncertainty = model(data)
                # Calcular pérdida física
                _, _, physics_loss = criterion(
                    (refined, simulated, uncertainty),
                    (target, data)
                )
                
                physics_metrics['signal_consistency'].append(
                    torch.mean((simulated - data)**2).item()
                )
                physics_metrics['uncertainty'].append(
                    torch.mean(uncertainty).item()
                )
                physics_metrics['physics_loss'].append(physics_loss.item())
            
    return {k: np.mean(v) for k, v in physics_metrics.items()}

def evaluate_model_comprehensive(model, test_loader, device, model_name, save_dir):
    """
    Evaluación comprehensiva del modelo incluyendo métricas estándar y físicas
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Métricas estándar
    metrics = {
        'mse': [], 'mae': [], 'r2': [], 
        'ssim': [], 'psnr': []
    }
    
    # Para visualizaciones
    sample_outputs = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc=f"Evaluando {model_name}")):
            data, target = data.to(device), target.to(device)
            
            if isinstance(model, PhysicsInformedWrapper):
                output, simulated, uncertainty = model(data)
            else:
                output = model(data)
            
            # Calcular métricas
            output_np = output.cpu().numpy()
            target_np = target.cpu().numpy()
            
            metrics['mse'].append(mean_squared_error(target_np.flatten(), output_np.flatten()))
            metrics['mae'].append(mean_absolute_error(target_np.flatten(), output_np.flatten()))
            metrics['r2'].append(r2_score(target_np.flatten(), output_np.flatten()))
            
            # Guardar algunas muestras para visualización
            if i < 5:
                if isinstance(model, PhysicsInformedWrapper):
                    sample_outputs.append({
                        'input': data[0,0].cpu().numpy(),
                        'target': target[0,0].cpu().numpy(),
                        'output': output[0,0].cpu().numpy(),
                        'simulated': simulated[0,0].cpu().numpy(),
                        'uncertainty': uncertainty[0,0].cpu().numpy()
                    })
                else:
                    sample_outputs.append({
                        'input': data[0,0].cpu().numpy(),
                        'target': target[0,0].cpu().numpy(),
                        'output': output[0,0].cpu().numpy()
                    })
    
    # Calcular métricas físicas si aplica
    if isinstance(model, PhysicsInformedWrapper):
        physics_metrics = evaluate_physics_metrics(model, test_loader, device)
        metrics.update(physics_metrics)
    
    # Calcular promedios
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k: np.std(v) for k, v in metrics.items()}
    
    # Guardar visualizaciones
    for i, sample in enumerate(sample_outputs):
        if isinstance(model, PhysicsInformedWrapper):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0,0].imshow(sample['input'])
            axes[0,0].set_title('Input')
            axes[0,1].imshow(sample['target'])
            axes[0,1].set_title('Ground Truth')
            axes[0,2].imshow(sample['output'])
            axes[0,2].set_title('Prediction')
            axes[1,0].imshow(sample['simulated'])
            axes[1,0].set_title('Simulated Signal')
            axes[1,1].imshow(sample['uncertainty'])
            axes[1,1].set_title('Uncertainty Map')
            axes[1,2].axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(sample['input'])
            axes[0].set_title('Input')
            axes[1].imshow(sample['target'])
            axes[1].set_title('Ground Truth')
            axes[2].imshow(sample['output'])
            axes[2].set_title('Prediction')
        
        plt.savefig(f'{save_dir}/sample_{i}_{model_name}.png')
        plt.close()
    
    # Guardar métricas
    with open(f'{save_dir}/metrics_{model_name}.txt', 'w') as f:
        for k in metrics.keys():
            f.write(f'{k}: {avg_metrics[k]:.6f} ± {std_metrics[k]:.6f}\n')
    
    return avg_metrics, std_metrics

def compare_models(models_dict, test_loader, device):
    """
    Compara múltiples modelos y genera visualizaciones comparativas
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'model_comparison_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # Evaluar cada modelo
    for model_name, model in models_dict.items():
        print(f"\nEvaluando {model_name}...")
        avg_metrics, std_metrics = evaluate_model_comprehensive(
            model, test_loader, device, model_name, save_dir
        )
        results[model_name] = {'avg': avg_metrics, 'std': std_metrics}
    
    # Crear visualizaciones comparativas
    metrics_to_plot = ['mse', 'mae', 'r2']
    if any(isinstance(model, PhysicsInformedWrapper) for model in models_dict.values()):
        metrics_to_plot.extend(['signal_consistency', 'physics_loss'])
    
    # Gráfico de barras comparativo
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        values = [results[model]['avg'].get(metric, 0) for model in results]
        errors = [results[model]['std'].get(metric, 0) for model in results]
        
        plt.bar(results.keys(), values, yerr=errors)
        plt.title(f'Comparison of {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/comparison_{metric}.png')
        plt.close()
    
    # Guardar resultados completos
    with open(f'{save_dir}/full_comparison.txt', 'w') as f:
        for model_name in results:
            f.write(f"\n{model_name}:\n")
            for metric in results[model_name]['avg']:
                avg = results[model_name]['avg'][metric]
                std = results[model_name]['std'][metric]
                f.write(f"{metric}: {avg:.6f} ± {std:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="Compare different models for photoacoustic image reconstruction.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar los modelos
    models = {}
    
    # UNet tradicional
    unet = UNet(in_channels=1, out_channels=1).to(device)
    unet_dir = max([d for d in os.listdir() if d.startswith("training_results_unet_")],
                   key=lambda x: os.path.getctime(x))
    unet.load_state_dict(torch.load(f'{unet_dir}/best_model.pth'))
    models['UNet'] = unet
    
    # UNet con atención
    # att_unet = AttentionUNet(in_channels=1, out_channels=1).to(device)
    # att_unet_dir = max([d for d in os.listdir() if d.startswith("training_results_attention_unet_")],
    #                    key=lambda x: os.path.getctime(x))
    # att_unet.load_state_dict(torch.load(f'{att_unet_dir}/best_model.pth'))
    # models['Attention_UNet'] = att_unet
    
    # Modelo físicamente informado
    phys_model = PhysicsInformedWrapper(AttentionUNet(in_channels=1, out_channels=1)).to(device)
    phys_dir = max([d for d in os.listdir() if d.startswith("finetuning_results_")],
                   key=lambda x: os.path.getctime(x))
    phys_model.load_state_dict(torch.load(f'{phys_dir}/best_finetuned_model.pth'))
    models['Physics_Informed'] = phys_model
    
    # Cargar datos de test
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Comparar modelos
    compare_models(models, test_loader, device)

if __name__ == "__main__":
    main()