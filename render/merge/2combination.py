
'''paste images on the background image, images with same id will be pasted on the same background image'''

import os
import random
from PIL import Image

# Paths to the folders containing the foreground and background images
foreground_folder = 'images'
background_folder = 'background'
output_folder = 'image_output_png'

# Get a list of all foreground and background files
foreground_files = os.listdir(foreground_folder)
background_files = os.listdir(background_folder)

# Filter out only the PNG files
foreground_files = [file for file in foreground_files if file.endswith('.png')]
background_files = [file for file in background_files if file.endswith('.png')]

# Group foreground images by their id (e.g., 0_p0_c0.png -> 0)
foreground_groups = {}
for file in foreground_files:
    file_id = file.split('_')[0]
    if file_id not in foreground_groups:
        foreground_groups[file_id] = []
    foreground_groups[file_id].append(file)


# Function to paste a foreground image onto a background image
def paste_images(foreground_path, background_path, output_path):
    # Open the foreground and background images
    foreground = Image.open(foreground_path)
    background = Image.open(background_path)

    # Resize the background image to match the size of the foreground image
    background = background.resize(foreground.size)

    # Paste the foreground image onto the background image
    background.paste(foreground, (0, 0), foreground)

    # Save the result
    background.save(output_path)


# Iterate over each group of foreground images
for file_id, files in foreground_groups.items():
    # Randomly select a background image
    background_file = random.choice(background_files)

    # Iterate over each foreground image in the group
    for file in files:
        foreground_path = os.path.join(foreground_folder, file)
        background_path = os.path.join(background_folder, background_file)
        output_path = os.path.join(output_folder, file)

        # Paste the images and save the result
        paste_images(foreground_path, background_path, output_path)

print("Pasting complete!")