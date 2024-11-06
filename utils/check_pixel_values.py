import os
from PIL import Image
import numpy as np

def check_image_pixels(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'): 
            image_path = os.path.join(directory, filename)
            
            with Image.open(image_path) as img:
                img_array = np.array(img)
                if not np.all(np.isin(img_array, [0, 34, 41])):
                    print(f"Image '{filename}' contains pixels with values other than 0, 34, or 41.")

check_image_pixels('/shares/CC_v_Val_FV_Gen3_all/SemsegCrosswalkRestrictedArea/labels')
