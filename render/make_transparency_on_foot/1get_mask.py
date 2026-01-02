import os
from PIL import Image


# Function to create a mask from a foreground image
def create_mask(foreground_path, mask_path):
    # Open the foreground image
    foreground = Image.open(foreground_path).convert("RGBA")

    # Create a new image for the mask with the same size as the foreground
    mask = Image.new("L", foreground.size, 0)

    # Get the alpha channel of the foreground image
    alpha = foreground.split()[-1]

    # Paste the alpha channel into the mask image
    mask.paste(alpha, (0, 0))

    # Save the mask image
    mask.save(mask_path)

input_dir = 'mask_origenal'
output_dir = 'mask'

for filename in os.listdir('mask_origenal'):
    foreground_path = os.path.join(input_dir, filename)
    mask_path = os.path.join(output_dir, filename.replace(".png", "_mask.png"))
    create_mask(foreground_path, mask_path)

