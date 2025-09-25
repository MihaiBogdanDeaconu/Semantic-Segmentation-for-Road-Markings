from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os

def denormalize(tensor, mean, std):
    """
    Reverses the normalization on a tensor.

    Args:
        tensor: The input tensor to denormalize.
        mean: The mean used for normalization.
        std: The standard deviation used for normalization.

    Returns:
        The denormalized tensor.
    """
    mean = np.array(mean)
    std = np.array(std)
    _mean = -mean / std
    _std = 1 / std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    """
    A callable class to denormalize a tensor, useful in torchvision transforms.
    """
    def __init__(self, mean, std):
        """Initializes with the mean and std used for normalization."""
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor):
        """Denormalizes the input tensor."""
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    """Sets the momentum for all BatchNorm2d layers in a model."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    """
    Sets all BatchNorm2d layers in a model to evaluation mode.
    This freezes the running mean and variance.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(path):
        os.mkdir(path)
