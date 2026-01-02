import os
from PIL import Image

# Path to the folder containing the foreground images
foreground_folder = 'images'
output_folder = 'mask_output'

# Get a list of all foreground files
foreground_files = os.listdir(foreground_folder)

# Filter out only the PNG files
foreground_files = [file for file in foreground_files if file.endswith('.png')]


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


# Iterate over each foreground image and create a mask
for file in foreground_files:
    foreground_path = os.path.join(foreground_folder, file)
    mask_path = os.path.join(output_folder, file)

    # Create and save the mask
    create_mask(foreground_path, mask_path)

print("Mask creation complete!")