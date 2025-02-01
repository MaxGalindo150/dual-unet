from scipy import stats
import numpy as np
import json
import argparse
from pathlib import Path

def calculate_confidence_intervals(pretrain_values, finetune_values, save_dir):
   metrics = {}
   for metric in ['MSE', 'MAE', 'SSIM']:
       pre_ci = stats.t.interval(0.95, len(pretrain_values[metric])-1, 
                               np.mean(pretrain_values[metric]), 
                               stats.sem(pretrain_values[metric]))
       
       fine_ci = stats.t.interval(0.95, len(finetune_values[metric])-1,
                                np.mean(finetune_values[metric]),
                                stats.sem(finetune_values[metric]))
       
       pre_bootstrap = stats.bootstrap((pretrain_values[metric],), np.mean,
                                    n_resamples=10000, confidence_level=0.95)
       
       fine_bootstrap = stats.bootstrap((finetune_values[metric],), np.mean,
                                     n_resamples=10000, confidence_level=0.95)
       
       metrics[metric] = {
           'pretrain': {
               'parametric_ci': [float(pre_ci[0]), float(pre_ci[1])],
               'bootstrap_ci': [float(pre_bootstrap.confidence_interval[0]), 
                              float(pre_bootstrap.confidence_interval[1])]
           },
           'finetune': {
               'parametric_ci': [float(fine_ci[0]), float(fine_ci[1])],
               'bootstrap_ci': [float(fine_bootstrap.confidence_interval[0]),
                              float(fine_bootstrap.confidence_interval[1])]
           }
       }
   
   save_results(metrics, save_dir)
   return metrics

def save_results(metrics, save_dir):
   Path(save_dir).mkdir(parents=True, exist_ok=True)
   
   with open(f'{save_dir}/confidence_intervals.txt', 'w') as f:
       f.write("Intervalos de Confianza (95%)\n")
       f.write("=" * 50 + "\n\n")
       
       for metric, results in metrics.items():
           f.write(f"\nMétrica: {metric}\n")
           f.write("-" * 20 + "\n")
           
           f.write("\nPretrain:\n")
           f.write(f"Paramétrico: [{results['pretrain']['parametric_ci'][0]:.6f}, "
                  f"{results['pretrain']['parametric_ci'][1]:.6f}]\n")
           f.write(f"Bootstrap: [{results['pretrain']['bootstrap_ci'][0]:.6f}, "
                  f"{results['pretrain']['bootstrap_ci'][1]:.6f}]\n")
           
           f.write("\nFine-tuning:\n")
           f.write(f"Paramétrico: [{results['finetune']['parametric_ci'][0]:.6f}, "
                  f"{results['finetune']['parametric_ci'][1]:.6f}]\n")
           f.write(f"Bootstrap: [{results['finetune']['bootstrap_ci'][0]:.6f}, "
                  f"{results['finetune']['bootstrap_ci'][1]:.6f}]\n")

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--pretrain_metrics', type=str, required=True)
   parser.add_argument('--finetune_metrics', type=str, required=True)
   parser.add_argument('--save_dir', type=str, required=True)
   
   args = parser.parse_args()
   
   with open(args.pretrain_metrics) as f:
       pretrain_metrics = json.load(f)
   with open(args.finetune_metrics) as f:
       finetune_metrics = json.load(f)
       
   calculate_confidence_intervals(pretrain_metrics, finetune_metrics, args.save_dir)

if __name__ == "__main__":
   main()