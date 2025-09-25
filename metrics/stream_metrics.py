import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    """
    Base class for streaming metrics.
    """
    def __init__(self):
        raise NotImplementedError

    def update(self, gt, pred):
        raise NotImplementedError

    def get_results(self):
        raise NotImplementedError

    def to_str(self, metrics):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class StreamSegMetrics(_StreamMetrics):
    """
    Computes and stores semantic segmentation metrics over a stream of data.

    This is useful for calculating metrics over an entire validation set
    batch by batch, without having to store all predictions in memory.
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        """
        Updates the confusion matrix with a new batch of predictions.

        Args:
            label_trues (numpy.ndarray): A batch of ground truth masks.
            label_preds (numpy.ndarray): A batch of predicted masks.
        """
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
    
    @staticmethod
    def to_str(results):
        """
        Converts the results dictionary to a human-readable string.
        """
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        """
        An efficient way to calculate the confusion matrix.
        """
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """
        Computes various segmentation metrics from the confusion matrix.
        
        Calculates Overall Accuracy, Mean Accuracy, Frequency Weighted IoU,
        and Mean IoU.

        Returns:
            (dict): A dictionary of computed metrics.
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        """
        Resets the confusion matrix to all zeros.
        """
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """
    A helper class to compute the average of a value over time.
    """
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        """Resets all tracked values."""
        self.book.clear()
    
    def reset(self, id):
        """Resets a specific value by its ID."""
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        """
        Updates a tracked value with a new reading.
        
        Args:
            id (str): The identifier for the value to track.
            val (float): The new value to add.
        """
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        """
        Gets the current average for a tracked value.

        Args:
            id (str): The identifier for the value.

        Returns:
            (float): The current average.
        """
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
