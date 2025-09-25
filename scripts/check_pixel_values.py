import os
from PIL import Image
import numpy as np

def check_image_pixels(directory):
    """
    Verifies that all label masks in a directory contain only expected pixel values.

    This is a quality-check script to ensure label files are correct for the
    competition format, which expects only values of 0, 34, and 41.

    Args:
        directory (str): The path to the directory containing the label masks.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.png'): 
            image_path = os.path.join(directory, filename)
            
            with Image.open(image_path) as img:
                img_array = np.array(img)
                if not np.all(np.isin(img_array, [0, 34, 41])):
                    print(f"Image '{filename}' contains pixels with values other than 0, 34, or 41.")

def main():
    """
    Runs the pixel check on a specified directory.
    """
    label_directory = '/path/to/labels'
    print(f"Checking pixel values in directory: {label_directory}")
    check_image_pixels(label_directory)
    print("Check complete.")

if __name__ == "__main__":
    main()
