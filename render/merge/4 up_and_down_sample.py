import os
from PIL import Image


def downsample_images(input_folder, output_folder, scale_factor=0.5):
    """
    Downsamples all JPG images in a folder by a given scale factor.

    Parameters:
        input_folder (str): Path to the input folder containing JPG images.
        output_folder (str): Path to the output folder for downsampled images.
        scale_factor (float): The factor by which to scale down the images (e.g., 0.5 for 50% size).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open the image
                with Image.open(input_path) as img:
                    # Calculate new dimensions
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    new_size = (new_width, new_height)

                    # Downsample the image using LANCZOS resampling
                    downsampled_img = img.resize(new_size, Image.Resampling.LANCZOS)

                    # Save the downsampled image
                    downsampled_img.save(output_path)
                    print(f"Saved downsampled image: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage:
input_folder = "image_harmonized"
output_folder = "augmented_images_jpg"
scale_factor = 0.7  # Scale down to 50%

downsample_images(input_folder, output_folder, scale_factor)