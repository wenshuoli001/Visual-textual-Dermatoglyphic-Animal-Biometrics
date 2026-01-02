import os
from PIL import Image, ImageEnhance

def darken_image(image_path, output_path, factor=0.5):
    # Open an image file
    with Image.open(image_path) as img:
        # Enhance the image brightness
        enhancer = ImageEnhance.Brightness(img)
        img_darken = enhancer.enhance(factor)
        # Save the darkened image
        img_darken.save(output_path)

def darken_images_in_directory(directory, factor=0.1):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            #output_path = os.path.join(directory, f"darkened_{filename}")
            output_path = os.path.join(directory, f"new/{filename}")
            darken_image(image_path, output_path, factor)

# Specify the directory containing the images
directory = "results3"

# Darken all PNG images in the directory
darken_images_in_directory(directory)