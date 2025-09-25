# Semantic Segmentation for Road Markings

This project implements a DeepLabV3+ model with a ResNet backbone to perform semantic segmentation on road markings, specifically identifying crosswalks and restricted areas. This solution was developed for the Bosch Semantic Segmentation Challenge and achieved a winning mIoU score of 87.


## Features

- **Model**: DeepLabV3+ with a ResNet101 or MobileNetV2 backbone.
- **Pre-training**: Utilizes weights pre-trained on the Cityscapes dataset for effective transfer learning.
- **Custom Dataset**: Includes a flexible data loader that reads image paths from a CSV file.
- **High Performance**: Achieved a score of 87 mIoU on the competition test set.
- **Functionality**: Provides clear scripts for both training and inference.

---

## Project Structure