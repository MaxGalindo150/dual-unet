import torch

from src.models.unet_model import UNet
from src.preprocess.preprocess_simulated_data import load_and_preprocess_data
from src.metrics.mae import calculate_batch_mae
def load_model_weights(model, checkpoint_path, device):
    """
    Carga los pesos del modelo manejando diferentes formatos de checkpoint.
    
    Args:
        model: Modelo PyTorch (UNet en este caso)
        checkpoint_path: Ruta al archivo de checkpoint
        device: Dispositivo donde cargar el modelo
    
    Returns:
        bool: True si la carga fue exitosa, False en caso contrario
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Caso 1: Checkpoint del fine-tuning (contiene unet_A_state_dict)
        if isinstance(checkpoint, dict) and 'unet_A_state_dict' in checkpoint:
            print("Detectado checkpoint de fine-tuning")
            model.load_state_dict(checkpoint['unet_A_state_dict'])
            return True
            
        # Caso 2: Checkpoint directo del modelo original
        elif isinstance(checkpoint, dict) and any(k.endswith(('.weight', '.bias')) for k in checkpoint.keys()):
            print("Detectado checkpoint de modelo base")
            model.load_state_dict(checkpoint)
            return True
            
        # Caso 3: Otro formato de checkpoint
        else:
            print("Formato de checkpoint no reconocido")
            print(f"Claves disponibles en el checkpoint: {checkpoint.keys()}")
            return False
            
    except Exception as e:
        print(f"Error al cargar el checkpoint: {str(e)}")
        return False

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
    
    # Cargar modelo
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # Intentar cargar los pesos
    if not load_model_weights(model, args.model_path, device):
        print("No se pudieron cargar los pesos del modelo. Abortando...")
        return
    
    # Cargar datos (ajusta según tu pipeline)
    _, _, test_loader = load_and_preprocess_data("simulated_data")
    
    # Calcular MSE
    mse_values = calculate_batch_mae(
        model, 
        test_loader, 
        device, 
        args.save_dir,
        args.model_name
    )
    
    print(f"\nCálculo de MAE completado para {args.model_name}")
    print(f"Resultados guardados en: {args.save_dir}")

if __name__ == "__main__":
    main()