from PIL import Image, ImageDraw
import os

def add_gradual_transparency(image_path, mask_path, output_path, transparency_strength):
    """
    Adds gradual transparency to a PNG image based on the foreground of a mask image.

    Args:
        image_path (str): Path to the PNG image.
        mask_path (str): Path to the mask image (white for foreground, black for background).
        output_path (str): Path to save the output image.
        transparency_strength (float): Starting transparency strength (0.0 - 1.0, where 1.0 is fully transparent).
    """
    # Open the PNG image and mask image
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale (L mode)

    # Ensure transparency strength is between 0.0 and 1.0
    transparency_strength = max(0.0, min(1.0, transparency_strength))

    # Find the bounding box of the white area in the mask
    bbox = mask.getbbox()  # Returns (left, upper, right, lower)
    if bbox is None:
        print("No white (foreground) area found in the mask.")
        return

    # Extract the foreground region
    left, upper, right, lower = bbox
    foreground_width = right - left
    foreground_height = lower - upper

    # Create a gradient image for the bounding box
    gradient = Image.new("L", (foreground_width, foreground_height), 0)
    draw = ImageDraw.Draw(gradient)

    # Fill gradient from the bottom (full transparency) to the top (no transparency)
    for y in range(foreground_height):
        alpha_value = int(255 * (1 - transparency_strength * (y / foreground_height)))
        draw.line([(0, y), (foreground_width, y)], fill=alpha_value)

    # Create a blank mask with the same size as the original mask
    adjusted_mask = Image.new("L", mask.size, 0)
    adjusted_mask.paste(gradient, (left, upper))

    # Extract alpha channel from the image
    r, g, b, alpha = image.split()

    # Combine the adjusted mask with the existing alpha channel
    new_alpha = Image.composite(adjusted_mask, alpha, mask)

    # Combine the RGB channels with the new alpha channel
    result = Image.merge("RGBA", (r, g, b, new_alpha))

    # Save the resulting image
    result.save(output_path)

'''set parameters'''
mask_dir = 'mask'
transparency_strength = 1  # Set transparency strength (0.0 for no transparency, 1.0 for full transparency)
'''======================================================================================================'''

#get the name of all the file folder in this directory
# Get the current directory
current_directory = os.getcwd()

# List all files and folders in the current directory
items = os.listdir(current_directory)
folder_list = []
# Print the names of the files and folders
for item in items:
    #print(item)
    #print(item[0])
    if item[0] == 'l' or item[0] == 'r':
        folder_list.append(item)
print(folder_list)

for folder in folder_list:
    fold_dir = folder + '/new'
    output_folder_dir = folder + '/output'
    if not os.path.exists(output_folder_dir):
        os.makedirs(output_folder_dir)

    for filename in os.listdir(fold_dir):
        image_path = os.path.join(fold_dir, filename)
        mask_path = os.path.join(mask_dir, folder + '_mask.png')
        output_path = os.path.join(output_folder_dir, filename)
        add_gradual_transparency(image_path, mask_path, output_path, transparency_strength)

# # Example usage
# image_path = "result0_result1.png"  # Replace with your PNG image path
# mask_path = "l_p0_c0_mask.png"  # Replace with your mask image path
# output_path = "output_image.png"  # Path to save the output image
# transparency_strength = 1  # Set transparency strength (0.0 for no transparency, 1.0 for full transparency)
#
# add_gradual_transparency(image_path, mask_path, output_path, transparency_strength)
