import numpy as np
from PIL import Image


def add_noise_to_image(image_path, noise_intensity):
    # Load the image
    img = Image.open(image_path)
    img = img.convert("RGBA")

    # Convert the image to numpy array
    data = np.array(img)

    # Generate random noise
    noise = np.random.normal(0, noise_intensity, data.shape)

    # Select non-transparent (opaque) pixels
    non_transparent = data[..., 3] > 0

    # Add noise to non-transparent pixels
    data[non_transparent] += noise[non_transparent].astype(int)

    # Clip values to be between 0 and 255
    data = np.clip(data, 0, 255)

    # Convert back to image
    noisy_img = Image.fromarray(data.astype(np.uint8))

    # Save the noisy image
    noisy_img.save("noisy_image.png")


# Call the function to add noise to the image
add_noise_to_image("b1.png", 10)

# Print a success message
print("Noise was successfully added to the image and saved as noisy_image.png")

