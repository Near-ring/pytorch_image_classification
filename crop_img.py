import os
import cv2
import numpy as np
from pathlib import Path

def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def crop_and_add_noise(input_folder, output_folder, crop_x=False, crop_y=True, stddev=25):
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Get the dimensions of the image
            h, w = image.shape[:2]
            
            # Calculate cropping bounds
            start_y, end_y = 0, h
            start_x, end_x = 0, w
            
            if crop_y:
                start_y = int(h * 0.33)
                end_y = int(h * 0.66)
            
            if crop_x:
                start_x = int(w * 0.33)
                end_x = int(w * 0.66)
            
            # Crop the image
            cropped_image = image[start_y:end_y, start_x:end_x]
            
            # Save the cropped image
            crop_output_path = os.path.join(output_folder, f'crop_{filename}')
            cv2.imwrite(crop_output_path, cropped_image)
            
            # Add Gaussian noise to the cropped image
            noisy_image = add_gaussian_noise(cropped_image, stddev=stddev)
            
            # Save the noisy image
            noisy_output_path = os.path.join(output_folder, f'crop_gn_{filename}')
            cv2.imwrite(noisy_output_path, noisy_image)

# Example usage:
input_folder = './'
output_folder = './a'
crop_and_add_noise(input_folder, output_folder, stddev=0.2)

