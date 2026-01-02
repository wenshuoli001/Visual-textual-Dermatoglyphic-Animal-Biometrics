import random

import cv2
import numpy as np
import os


'''define nosie and blur functions'''

def add_gaussian_noise(image, strength=0.1):
    """
    Adds Gaussian noise to an image with strength control.

    Parameters:
        image (numpy.ndarray): Input image (BGR format).
        strength (float): Strength of the noise (0 to 1). Higher means more noise.

    Returns:
        numpy.ndarray: Noisy image.
    """
    # Ensure strength is in the valid range
    strength = np.clip(strength, 0, 1)

    # Compute standard deviation based on strength
    stddev = strength * 255  # Scale noise intensity to image range (0-255)
    mean = 0

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)

    # Add noise to the image
    noisy_image = image.astype(np.float32) + noise

    # Clip to ensure pixel values stay within valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def add_motion_blur(image, strength=10, angle=0):
    """
    Adds motion blur to an image.

    Parameters:
        image (numpy.ndarray): Input image (BGR format).
        strength (int): Strength of the blur (length of the motion blur kernel).
        angle (float): Angle of the motion blur in degrees (0 for horizontal).

    Returns:
        numpy.ndarray: Motion blurred image.
    """
    # Ensure strength is a positive integer
    strength = max(1, int(strength))

    # Create a motion blur kernel
    kernel = np.zeros((strength, strength), dtype=np.float32)
    center = strength // 2

    # Set the line in the kernel to create motion blur
    if angle == 0:  # Horizontal motion blur
        kernel[center, :] = 1.0
    elif angle == 90:  # Vertical motion blur
        kernel[:, center] = 1.0
    else:  # Arbitrary angle
        for i in range(strength):
            x = int(center + (i - center) * np.cos(np.radians(angle)))
            y = int(center + (i - center) * np.sin(np.radians(angle)))
            if 0 <= x < strength and 0 <= y < strength:
                kernel[y, x] = 1.0

    kernel /= kernel.sum()  # Normalize the kernel

    # Apply the kernel to the image
    blurred_image = cv2.filter2D(image, -1, kernel)

    return blurred_image



'''define parameters'''
noise_strength = 0.07  # Adjust noise strength (0 for no noise, 1 for very noisy)
blur_strength = 3  # Strength of the blur (length of the kernel)
blur_angle = 0     # Angle of the blur in degrees

input_dir = 'augmented_images_jpg'
output_dir = 'augmented_images_jpg_noise_blur'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        image = cv2.imread(input_path)
        #add noise or blur for 20% of the images
        if random.random() < 0.2:
            image = add_gaussian_noise(image, noise_strength)
            print('noise added to', filename )
        if random.random() < 0.1:
            image = add_motion_blur(image, blur_strength, blur_angle)
            print('blur added to', filename)
        cv2.imwrite(output_path, image)

