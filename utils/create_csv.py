import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def main():
    path = r"C:\Users\DEM7CLJ\Documents\Semseg\Painted-Symbols-and-Restricted-Zone-Segmentation\train_rgb\splited_labels\background_restricted_crosswalk"
    image_names = os.listdir(path)

    image_paths = [os.path.join(path.split('\\splited_labels')[0], 'images', image_name) for image_name in image_names if not image_name == 'Thumbs.db']

    stages = ['train' if random.random() < 0.8 else 'eval' for _ in image_paths]

    df = pd.DataFrame({'path': image_paths, 'split': stages})

    df.to_csv(r"C:\Users\DEM7CLJ\Documents\Semseg\Painted-Symbols-and-Restricted-Zone-Segmentation\dataset.csv", index=False)

    print("DataFrame created and saved to 'image_stages.csv'.")

if __name__ == "__main__":
    main()
