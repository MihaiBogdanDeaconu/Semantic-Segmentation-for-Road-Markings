from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """
    Implements a polynomial learning rate decay scheduler.

    The learning rate is decayed from a base rate following a polynomial function
    of the current iteration. This is a common scheduler for segmentation tasks.
    """
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        """
        Initializes the PolyLR scheduler.

        Args:
            optimizer: The optimizer for which to schedule the learning rate.
            max_iters: The total number of training iterations.
            power: The exponent of the polynomial.
            last_epoch: The index of the last epoch.
            min_lr: A lower bound on the learning rate.
        """
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculates the learning rate for the current epoch."""
        return [max(base_lr * (1 - self.last_epoch / self.max_iters)**self.power, self.min_lr)
                for base_lr in self.base_lrs]
