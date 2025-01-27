from PIL import Image
import os

def convert_to_binary(image_path, output_path, threshold=128):
    # Abre la imagen
    image = Image.open(image_path)
    # Convierte la imagen a escala de grises
    gray_image = image.convert('L')
    # Aplica el umbral para convertir a imagen binaria
    binary_image = gray_image.point(lambda x: 255 if x > threshold else 0, '1')
    # Guarda la imagen binaria
    binary_image.save(output_path)

def process_directory(input_dir, output_dir, threshold=128):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.ppm'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_binary.ppm')
            convert_to_binary(input_path, output_path, threshold)

if __name__ == "__main__":
    input_directory = '/home/mgalindo/max/maestria/tesis/w_deep/data/all_images'
    output_directory = '/home/mgalindo/max/maestria/tesis/w_deep/data/binary_images'
    process_directory(input_directory, output_directory)