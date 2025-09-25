import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function.

    Focal Loss is an enhancement of Cross-Entropy Loss that down-weights the
    loss assigned to well-classified examples, helping the model focus on
    hard, misclassified examples.
    """
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        """
        Initializes the FocalLoss module.

        Args:
            alpha (int): A balancing factor for the loss.
            gamma (int): The focusing parameter. Higher values give more weight
                         to hard-to-classify examples.
            size_average (bool): If True, the loss is averaged over all samples.
            ignore_index (int): A class index to ignore in the loss calculation.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        """Calculates the focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
