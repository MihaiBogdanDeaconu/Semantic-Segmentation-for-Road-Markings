import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from scripts import data_load
from collections import namedtuple

class PaintedSymbols(data.Dataset):
    """
    Custom dataset for the Bosch Road Markings Challenge.
    This class loads image and label paths from a CSV file and applies
    transformations. It also handles mapping the competition's label values
    (e.g., 34, 41) to simple training IDs (e.g., 1, 2).
    """
    PaintedSymbolsClass = namedtuple(
        "PaintedSymbolsClass",
        ["name", "id", "color"],
    )
    classes = [
        PaintedSymbolsClass("background", 0, (0, 0, 0)),
        PaintedSymbolsClass("crosswalk", 1, (0, 255, 0)),
        PaintedSymbolsClass("restricted_area", 2, (255, 255, 0)),
    ]

    train_id_to_color = {cls.id: cls.color for cls in classes}
    
    def __init__(self, csv_path, split="train", transform=None):
        """
        Initializes the dataset.

        Args:
            csv_path (str): Path to the CSV file with image paths and splits.
            split (str): The dataset split to use ('train' or 'eval').
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.split = split
        self.transform = transform
        self.images, self.targets = data_load.load_from_csv(csv_path, self.split)
        print(f"Loaded {len(self.images)} images for split '{self.split}'")

    @classmethod
    def decode_target(cls, target):
        """
        Converts a segmentation mask with training IDs to a color image for visualization.

        Args:
            target (numpy.ndarray): A numpy array with values in {0, 1, 2}.
        
        Returns:
            (numpy.ndarray): A colorized RGB image.
        """
        rgb_mask = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        for train_id, color in cls.train_id_to_color.items():
            rgb_mask[target == train_id] = color
        return rgb_mask

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding target mask at a given index.

        This method also maps the original label values (34 for crosswalk, 41 for
        restricted area) to the training IDs (1 and 2, respectively).

        Args:
            index (int): The index of the data sample to retrieve.

        Returns:
            A tuple of (torch.Tensor, torch.Tensor): The image and its target mask.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transform:
            image, target = self.transform(image, target)
        
        target = torch.where(target == 34, 1, target)
        target = torch.where(target == 41, 2, target)
        target = torch.where((target != 0) & (target != 1) & (target != 2), 0, target)
        
        return image, target

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)
