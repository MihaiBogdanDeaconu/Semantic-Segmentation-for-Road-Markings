import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from utils import data_load


class PaintedSymbols(data.Dataset):
    """Painted Symbols

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    PaintedSymbolsClass = namedtuple(
        "PaintedSymbolsClass",
        [
            "name",
            "id",
            "color",
        ],
    )
    classes = [
        PaintedSymbolsClass("background", 0, (255, 0, 0)),
        PaintedSymbolsClass("crosswalk", 34, (0, 255, 0)),
        PaintedSymbolsClass("restricted_area", 41, (0, 0, 255)),
    ]

    ids = [0, 34, 41]

    id_to_color = {
        0: (255, 0, 0),
        34: (0, 255, 0),
        41: (0, 0, 255),
    }

    def __init__(self, split="train", target_type="semantic", transform=None):
        self.target_type = target_type
        self.split = split
        self.transform = transform
        self.images, self.targets = data_load.load_from_csv(r"/home/dem7clj/SemsegChallenge/Painted-Symbols-and-Restricted-Zone-Segmentation/datasetWithBackground.csv", self.split)

    @classmethod
    def decode_target(cls, target):
        return cls.id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index])
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)

        target = torch.where(target == 0, 0, target)
        target = torch.where(target == 34, 1, target)
        target = torch.where(target == 41, 2, target)
        return image, target

    def __len__(self):
        return len(self.targets)

    def _load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
