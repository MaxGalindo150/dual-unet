import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.draw import polygon, disk
from types import SimpleNamespace
import pickle
from tqdm import tqdm

def generate_random_shape(W, center_y, center_x, shape_type='random'):
    """Genera una forma aleatoria en la posición especificada"""
    if shape_type == 'random':
        shape_type = np.random.choice(['point', 'triangle', 'rectangle', 'circle'])
    
    if shape_type == 'point':
        s = np.random.randint(2, 4)  # Tamaño aleatorio del punto
        rr, cc = polygon([center_y, center_y, center_y + s, center_y + s],
                        [center_x, center_x + s, center_x + s, center_x])
        intensity = np.random.uniform(0.75, 1.0)
    
    elif shape_type == 'triangle':
        s = np.random.randint(8, 15)  # Tamaño aleatorio del triángulo
        rr, cc = polygon([center_y, center_y + s, center_y],
                        [center_x, center_x, center_x + s])
        intensity = np.random.uniform(0.5, 0.75)
    
    elif shape_type == 'rectangle':
        s = np.random.randint(6, 12)  # Tamaño aleatorio del rectángulo
        rr, cc = polygon([center_y, center_y, center_y + s, center_y + s],
                        [center_x, center_x + s, center_x + s, center_x])
        intensity = np.random.uniform(0.25, 0.5)
    
    else:  # circle
        s = np.random.randint(6, 12)  # Radio aleatorio del círculo
        rr, cc = disk((center_y, center_x), s, shape=W.shape)
        intensity = np.random.uniform(0.1, 0.25)
    
    # Asegurarse de que los índices están dentro de los límites
    mask = (rr < W.shape[0]) & (cc < W.shape[1])
    rr, cc = rr[mask], cc[mask]
    
    W[rr, cc] = intensity
    return W

def generate_sample(Nz=128, Nx=128, num_shapes=4):
    """Genera una muestra con múltiples formas aleatorias"""
    W = np.zeros((Nz, Nx))
    
    # Generar posiciones aleatorias evitando los bordes
    margin = 20
    positions_y = np.random.randint(margin, Nz-margin, num_shapes)
    positions_x = np.random.randint(margin, Nx-margin, num_shapes)
    
    # Generar formas aleatorias
    for i in range(num_shapes):
        W = generate_random_shape(W, positions_y[i], positions_x[i])
    
    return W

def nextpow2(x):
    """Encuentra la siguiente potencia de 2"""
    return int(np.ceil(np.log2(np.abs(x))))

def generate_photoacoustic_measurement(W, c0=1, sigma=1):
    """Genera mediciones fotoacústicas a partir de W"""
    Nz, Nx = W.shape
    Nt = 128
    
    # Construcción de la cuadrícula
    Z = np.zeros(W.shape)
    ZP_col = np.zeros((Nz, 2**(nextpow2(2*Nx) + 1) - 2*Nx))
    ZP_row = np.zeros((2**(nextpow2(2*Nz) + 1) - 2*Nz, 2**(nextpow2(2*Nx) + 1)))
    
    # Matriz P
    P = np.block([
        [Z, W, ZP_col],
        [Z, Z, ZP_col],
        [ZP_row]
    ])
    
    Ly, Lx = P.shape
    
    # Crear vectores kx y ky
    kx = (np.arange(Lx) / Lx) * np.pi
    ky = (np.arange(Ly) / Ly) * np.pi
    
    # Crear la matriz de frecuencia
    f = np.zeros((Lx, Ly))
    for kxi in range(Lx):
        for kyi in range(Ly):
            f[kxi, kyi] = np.sqrt(kx[kxi]**2 + ky[kyi]**2)
    
    # Generar mediciones
    Pdet = np.zeros((P.shape[0], len(f)))
    P_hat = dct(dct(P.T, norm='ortho').T, norm='ortho')
    
    for t in range(P.shape[0]):
        cost = np.cos(c0 * f * t)
        Pcos = P_hat * cost.T
        Pt = idct(idct(Pcos.T, norm='ortho').T, norm='ortho') / 3
        Pdet[t, :] = Pt[0, :]
    
    # Agregar ruido gaussiano
    noise = np.random.normal(0, sigma * np.std(Pdet), Pdet.shape)
    P_surf = Pdet + noise
    
    return P_surf

def main():
    
    parser = argparse.ArgumentParser(description="Generate simulated photoacoustic data.")
    parser.add_argument('--num_samples', type=int, default=2500,
                        help="Number of samples to generate. (default: 2500)")
    args = parser.parse_args()
    # Crear directorio para los datos si no existe
    os.makedirs('simulated_data', exist_ok=True)
    
    # Parámetros
    num_samples = args.num_samples
    Nz = 128  # Resolución en profundidad
    Nx = 128  # Resolución espacial
    
    print("Generando datos simulados...")
    for i in tqdm(range(num_samples)):
        # Generar ground truth
        W = generate_sample(Nz, Nx)
        
        # Generar medición fotoacústica
        P_surf = generate_photoacoustic_measurement(W)
        
        # Guardar los datos
        sample_data = {
            'ground_truth': W,
            'measurement': P_surf
        }
        
        # Guardar en formato pickle
        filename = f'simulated_data/sample_{i:04d}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(sample_data, f)
        
        # Opcional: guardar visualizaciones
        if i < 5:  # Guardar solo las primeras 5 muestras como imágenes
            plt.figure(figsize=(12, 5))
            
            plt.subplot(121)
            plt.imshow(W.T, aspect='auto', cmap='gray')
            plt.title('Ground Truth')
            plt.colorbar()
            
            plt.subplot(122)
            plt.imshow(P_surf.T, aspect='auto')
            plt.title('Photoacoustic Measurement')
            plt.colorbar()
            
            plt.savefig(f'simulated_data/sample_{i:04d}_viz.png')
            plt.close()

if __name__ == "__main__":
    main()