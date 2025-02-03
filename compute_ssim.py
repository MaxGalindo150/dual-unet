import torch

from src.models.unet_model import UNet
from src.preprocess.preprocess_simulated_data import load_and_preprocess_data
from src.metrics.ssim import calculate_batch_ssim

def main():
    """
    Función principal para calcular MSE de un modelo.
    
    Uso:
    python calculate_mse.py --model_path path/to/model --model_name model_version
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate MSE for reconstruction model")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument('--model_name', type=str, required=True,
                       help="Identifier name for the model")
    parser.add_argument('--save_dir', type=str, default='mse_results',
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar modelo (ajusta según tu arquitectura)
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Cargar datos (ajusta según tu pipeline)
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Calcular MSE
    mse_values = calculate_batch_ssim(
        model, 
        test_loader, 
        device, 
        args.save_dir,
        args.model_name
    )
    
    print(f"\nCálculo de SSIM completado para {args.model_name}")
    print(f"Resultados guardados en: {args.save_dir}")

if __name__ == "__main__":
    main()