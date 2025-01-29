import cv2
import numpy as np
import os
from scipy.ndimage import rotate

from tqdm import tqdm

def augment_image(image, seed=None):
    # Configurar semilla única por augmentación
    rng = np.random.default_rng(seed)
    
    h, w = image.shape[:2]
    
    # 1. Transformaciones geométricas (siempre aplicadas)
    # ---------------------------------------------------
    # Rotación con ángulo y centro variables
    angle = rng.uniform(-30, 30)
    center_x = rng.uniform(0.4*w, 0.6*w)
    center_y = rng.uniform(0.4*h, 0.6*h)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    augmented = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    
    # Volteos independientes
    if rng.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    if rng.random() > 0.5:
        augmented = cv2.flip(augmented, 0)
    
    # 2. Transformaciones de escala (80% de probabilidad)
    # ---------------------------------------------------
    if rng.random() < 0.8:
        scale = rng.choice([rng.uniform(0.5, 0.8), rng.uniform(1.2, 1.5)], 
                          p=[0.5, 0.5])
        new_h, new_w = int(h*scale), int(w*scale)
        scaled = cv2.resize(augmented, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if scale < 1:  # Padding aleatorio
            pad_h = h - new_h
            pad_w = w - new_w
            augmented = cv2.copyMakeBorder(scaled,
                                          rng.integers(0, pad_h),
                                          pad_h - rng.integers(0, pad_h),
                                          rng.integers(0, pad_w),
                                          pad_w - rng.integers(0, pad_w),
                                          cv2.BORDER_REFLECT_101)
        else:  # Crop aleatorio
            start_h = rng.integers(0, new_h - h)
            start_w = rng.integers(0, new_w - w)
            augmented = scaled[start_h:start_h+h, start_w:start_w+w]
    
    # 3. Transformaciones de color (combinaciones aleatorias)
    # --------------------------------------------------------
    color_transforms = []
    
    # Brillo/Contraste (70% de probabilidad)
    if rng.random() < 0.7:
        alpha = rng.uniform(0.5, 1.8)
        beta = rng.uniform(-30, 30)
        color_transforms.append(lambda x: cv2.convertScaleAbs(x, alpha=alpha, beta=beta))
    
    # Gamma (50% de probabilidad)
    if rng.random() < 0.5:
        gamma = rng.uniform(0.4, 2.0)
        table = np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)]).astype("uint8")
        color_transforms.append(lambda x: cv2.LUT(x, table))
    
    # Modificaciones HSV (solo color, 60% de probabilidad)
    if len(image.shape) == 3 and rng.random() < 0.6:
        h_shift = rng.uniform(-0.1, 0.1)
        s_scale = rng.uniform(0.6, 1.6)
        v_scale = rng.uniform(0.6, 1.6)
        
        def hsv_transform(x):
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + h_shift*180) % 180
            hsv[..., 1] = np.clip(hsv[..., 1]*s_scale, 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2]*v_scale, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        color_transforms.append(hsv_transform)
    
    # Aplicar transformaciones de color en orden aleatorio
    rng.shuffle(color_transforms)
    for transform in color_transforms:
        augmented = transform(augmented)
    
    # 4. Distorsiones (aplicar 1-2 por imagen)
    # ----------------------------------------
    distortions = rng.choice([
        'noise', 
        'blur', 
        'compression', 
        'none'
    ], size=rng.integers(1, 3), replace=False, p=[0.4, 0.3, 0.2, 0.1])
    
    for dist in distortions:
        if dist == 'noise':
            noise_type = rng.choice(['gaussian', 'speckle', 'salt_pepper'], 
                                   p=[0.5, 0.3, 0.2])
            if noise_type == 'gaussian':
                sigma = rng.uniform(0, 25)
                noise = rng.normal(0, sigma, augmented.shape).astype(np.int16)
                augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif noise_type == 'speckle':
                noise = augmented * rng.normal(0, 0.2, augmented.shape)
                augmented = np.clip(augmented + noise.astype(np.int16), 0, 255).astype(np.uint8)
            else:  # Salt & Pepper
                amount = rng.uniform(0.001, 0.01)
                augmented = cv2.medianBlur(augmented, 3)
        
        elif dist == 'blur':
            kernel_size = rng.choice([3, 5, 7])
            blur_type = rng.choice(['gaussian', 'median', 'bilateral'])
            if blur_type == 'gaussian':
                augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
            elif blur_type == 'median':
                augmented = cv2.medianBlur(augmented, kernel_size)
            else:
                augmented = cv2.bilateralFilter(augmented, kernel_size, 75, 75)
        
        elif dist == 'compression':
            quality = rng.integers(10, 90)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', augmented, encode_param)
            augmented = cv2.imdecode(encimg, 1)
    
    return augmented

def main():
    input_dir = 'data/vessels'
    output_dir = 'data/augmented_vessels'
    num_augmentations = 125  # Número de imágenes aumentadas a generar por cada imagen original
    
    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Listar imágenes en el directorio de entrada
    image_files = os.listdir(input_dir)
    
    # Iterar sobre las imágenes con barra de progreso
    for img_idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        # Cargar imagen
        image = cv2.imread(os.path.join(input_dir, image_file), cv2.IMREAD_GRAYSCALE)
        
        # Generar múltiples imágenes aumentadas
        for aug_idx in range(num_augmentations):
            # Usar semilla única basada en índice de imagen y aumento
            seed = img_idx * 1000 + aug_idx  
            augmented = augment_image(image, seed=seed)
            
            # Guardar con nombre único
            cv2.imwrite(f'{output_dir}/img_{img_idx}_aug_{aug_idx}.png', augmented)
    
    print('Augmentation finished!')

            
if __name__ == "__main__":
    main()
        