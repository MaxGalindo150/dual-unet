import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.draw import polygon, disk
import pickle
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
import time
import uuid

# ----------------------------------------------------------------
# Configurar el método de inicio de los procesos como 'spawn'
# ----------------------------------------------------------------
set_start_method('spawn', force=True)  

# ----------------------------------------------------------------
# Inicialización segura del generador de números aleatorios por worker
# ----------------------------------------------------------------
def init_worker():
    """Inicializa un generador de números aleatorios único para cada worker"""
    seed = (os.getpid() + int(time.time())) % 2**32
    np.random.seed(seed)

def generate_random_shape(W, center_y, center_x):
    """Genera una forma aleatoria en la posición especificada"""
    shape_type = np.random.choice(['point', 'triangle', 'rectangle', 'circle'])
    
    # Parámetros aleatorios específicos para cada forma
    if shape_type == 'point':
        s = np.random.randint(2, 4)
        rr, cc = polygon([center_y, center_y, center_y + s, center_y + s],
                        [center_x, center_x + s, center_x + s, center_x])
        intensity = np.random.uniform(0.75, 1.0)
    
    elif shape_type == 'triangle':
        s = np.random.randint(8, 15)
        rr, cc = polygon([center_y, center_y + s, center_y],
                        [center_x, center_x, center_x + s])
        intensity = np.random.uniform(0.5, 0.75)
    
    elif shape_type == 'rectangle':
        s = np.random.randint(6, 12)
        rr, cc = polygon([center_y, center_y, center_y + s, center_y + s],
                        [center_x, center_x + s, center_x + s, center_x])
        intensity = np.random.uniform(0.25, 0.5)
    
    else:  # circle
        s = np.random.randint(6, 12)
        rr, cc = disk((center_y, center_x), s, shape=W.shape)
        intensity = np.random.uniform(0.1, 0.25)
    
    # Asegurar que los índices estén dentro de los límites
    mask = (rr < W.shape[0]) & (cc < W.shape[1])
    rr, cc = rr[mask], cc[mask]
    
    W[rr, cc] = intensity
    return W

def generate_sample(Nz=128, Nx=128, num_shapes=4):
    """Genera una muestra con múltiples formas aleatorias"""
    W = np.zeros((Nz, Nx))
    margin = 20
    
    # Generar posiciones aleatorias únicas
    positions_y = np.random.randint(margin, Nz - margin, num_shapes)
    positions_x = np.random.randint(margin, Nx - margin, num_shapes)
    
    for i in range(num_shapes):
        W = generate_random_shape(W, positions_y[i], positions_x[i])
    
    return W

def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))

def generate_photoacoustic_measurement(W):
    sigma = 1.0
    c0 = 1.0
    Nz, Nx = W.shape
    Z = np.zeros(W.shape)
    ZP_col = np.zeros((Nz, 2**(nextpow2(2*Nx) + 1) - 2*Nx))
    ZP_row = np.zeros((2**(nextpow2(2*Nz) + 1) - 2*Nz, 2**(nextpow2(2*Nx) + 1)))
    
    P = np.block([[Z, W, ZP_col], [Z, Z, ZP_col], [ZP_row]])
    Ly, Lx = P.shape
    
    kx = (np.arange(Lx) / Lx) * np.pi
    ky = (np.arange(Ly) / Ly) * np.pi
    
    f = np.zeros((Lx, Ly))
    for kxi in range(Lx):
        for kyi in range(Ly):
            f[kxi, kyi] = np.sqrt(kx[kxi]**2 + ky[kyi]**2)
    
    Pdet = np.zeros((P.shape[0], len(f)))
    P_hat = dct(dct(P.T, norm='ortho').T, norm='ortho')
    
    for t in range(P.shape[0]):
        cost = np.cos(c0 * f * t)
        Pcos = P_hat * cost.T
        Pt = idct(idct(Pcos.T, norm='ortho').T, norm='ortho') / 3
        Pdet[t, :] = Pt[0, :]
    
    noise = np.random.normal(0, sigma * np.std(Pdet), Pdet.shape)
    P_surf = Pdet + noise
    return P_surf

def process_sample(args):
    """Genera y guarda una muestra única"""
    idx, Nz, Nx = args
    # Generar una semilla única usando el índice y el tiempo actual
    np.random.seed((idx + int(time.time())) % 2**32)
    
    W = generate_sample(Nz, Nx)
    P_surf = generate_photoacoustic_measurement(W)
    
    sample_data = {'ground_truth': W, 'measurement': P_surf}
    filename = f'simulated_data/sample_{idx:04d}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(sample_data, f)
    
    if idx < 5:
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(W.T, cmap='gray')
        plt.title(f'Ground Truth (Sample {idx})')
        plt.subplot(122)
        plt.imshow(P_surf.T)
        plt.title(f'Measurement (Sample {idx})')
        plt.savefig(f'simulated_data/sample_{idx:04d}_viz.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate simulated photoacoustic data.")
    parser.add_argument('--num_samples', type=int, default=2500)
    parser.add_argument('--num_workers', type=int, default=12)  # 12 workers para 24 hilos
    args = parser.parse_args()
    
    os.makedirs('simulated_data', exist_ok=True)
    task_args = [(i, 128, 128) for i in range(args.num_samples)]
    
    print("Generando datos simulados...")
    with Pool(
        args.num_workers,
        initializer=init_worker  # Inicializa cada worker con una semilla única
    ) as pool:
        list(tqdm(pool.imap(process_sample, task_args), total=args.num_samples))

if __name__ == "__main__":
    main()