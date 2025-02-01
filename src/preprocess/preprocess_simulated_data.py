import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class PhotoacousticDataset(Dataset):
    def __init__(self, measurements, ground_truths):
        self.measurements = torch.FloatTensor(measurements)
        self.ground_truths = torch.FloatTensor(ground_truths)
    
    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
        return self.measurements[idx], self.ground_truths[idx]

def load_and_preprocess_data(data_dir, batch_size=32):
    """
    Carga y preprocesa los datos fotoacústicos para entrenamiento
    """
    # Listas para almacenar datos
    measurements = []
    ground_truths = []
    
    # Cargar todos los archivos .pkl
    print("Loading data...")
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), 'rb') as f:
                data = pickle.load(f)
                
                # Extraer datos
                measurement = data['measurement']
                ground_truth = data['ground_truth']
                
                # Redimensionar mediciones si es necesario (asegurarse de que todas tengan el mismo tamaño)
                from skimage.transform import resize
                
                if measurement.shape != (128, 128):
                    measurement = resize(measurement, (128, 128), anti_aliasing=True)
                if ground_truth.shape != (128, 128):
                    ground_truth = resize(ground_truth, (128, 128), anti_aliasing=True)
                
                
                measurements.append(measurement)
                ground_truths.append(ground_truth)
    
    # Convertir a arrays numpy
    measurements = np.array(measurements)
    ground_truths = np.array(ground_truths)
    
    # Normalización
    print("Normalizing data...")
    measurements = (measurements - np.mean(measurements)) / np.std(measurements)
    ground_truths = (ground_truths - np.min(ground_truths)) / (np.max(ground_truths) - np.min(ground_truths))
    
    # Añadir dimensión de canal (necesaria para CNNs)
    measurements = measurements[:, np.newaxis, :, :]
    ground_truths = ground_truths[:, np.newaxis, :, :]
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        measurements, ground_truths, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42
    )
    
    # Crear datasets de PyTorch
    train_dataset = PhotoacousticDataset(X_train, y_train)
    val_dataset = PhotoacousticDataset(X_val, y_val)
    test_dataset = PhotoacousticDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    return train_loader, val_loader, test_loader

def save_preprocessed_data(data_dir, output_dir='preprocessed_data'):
    """
    Guarda los datos preprocesados
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar y preprocesar datos
    train_loader, val_loader, test_loader = load_and_preprocess_data(data_dir)
    
    # Guardar los dataloaders
    print("Guardando datos preprocesados...")
    preprocessed_data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }
    
    with open(os.path.join(output_dir, 'preprocessed_data.pkl'), 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"Datos preprocesados guardados en {output_dir}")

if __name__ == "__main__":
    data_dir = "simulated_data"  # Directorio con los datos originales
    save_preprocessed_data(data_dir)