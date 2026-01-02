import os
from PIL import Image

# Path to the folder containing the PNG images
input_folder = 'mask_output'
output_folder = 'masks_jpg'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to convert PNG to JPEG
def convert_to_jpeg(input_path, output_path):
    image = Image.open(input_path).convert("RGB")
    image.save(output_path, "JPEG")

# Iterate over each file in the input folder
for file in os.listdir(input_folder):
    if file.endswith('.png'):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.png', '.jpg'))
        convert_to_jpeg(input_path, output_path)

print("Conversion complete!")