import torch
from torch.nn import functional as F
import numpy as np

# from pyhessian import hessian
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt

# from sklearn.metrics import auc, average_precision_score, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.metrics import f1_score
from torchmetrics import Precision, Recall

# from torchmetrics.functional import auc as AUC

# from torchmetrics.functional import f1_score
from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)


class Metrics:
    def get_binary_accuracy(self, logit, y, threshold=0):
        y_hat = (logit >= threshold).long().reshape(-1)
        y = y.reshape(-1)
        return (y_hat == y).long().sum().item() * 1.0 / len(y_hat)

    def get_accuracy(self, y_hat, y, criterion):
        if criterion == "ce":
            return (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
        elif criterion == "bce":
            return self.get_binary_accuracy(y_hat, y)

    @staticmethod
    def get_binary_metric(y_hat, y):
        torch.save([y, y_hat], "temp.pt")

        if len(y_hat.shape) == 2:
            y_hat = y_hat.argmax(dim=1).detach().cpu()
        y = y.detach().cpu()
        y_hat = y_hat.detach().cpu()
        try:
            auc = AUC(y, y_hat)
            # f1 = (
            #     np.asarray(
            #         [f1_score(y > x, y_hat) for x in np.linspace(0.1, 1, num=10) if (y > x).any() and (y < x).any()]
            #     )
            #     .max()
            #     .item()
            # )
            f1 = f1_score(y, y_hat)
        except ValueError:
            auc = 5
            f1 = 5
        rec = Recall(num_classes=2, average="micro")
        prec = Precision(num_classes=2, average="micro")
        recall = rec(y, y_hat)
        precision = prec(y, y_hat)
        return auc, f1, precision, recall

    @staticmethod
    def calc_pcc(tensor1, tensor2, **kwargs):
        # permute if channels are first
        if not (isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray)):
            array1 = torch_to_numpy(tensor1)
            array2 = torch_to_numpy(tensor2)

        return pearsonr(array1.reshape(-1), array2.reshape(-1), **kwargs)[0]

    @staticmethod
    def calc_ssim(tensor1, tensor2, **kwargs):
        if not (isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray)):
            if len(tensor1.shape) == 4:
                array1 = tensor1.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
                array2 = tensor2.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()

            else:
                array1 = tensor1.contiguous().squeeze().detach().cpu().numpy()
                array2 = tensor2.contiguous().squeeze().detach().cpu().numpy()

        max_v = max(array1.max(), array2.max())
        min_v = min(array1.min(), array2.min())

        # check for 3 channel image
        if len(array1.shape) == 3:
            kwargs["multichannel"] = True

        return structural_similarity(array1, array2, data_range=max_v - min_v, **kwargs)

    @staticmethod
    def calc_cossim(tensor1, tensor2, **kwargs):
        tensor1 = tensor1.flatten(start_dim=1)
        tensor2 = tensor2.flatten(start_dim=1)

        return F.cosine_similarity(tensor1, tensor2)


def torch_to_numpy(tensor):
    if len(tensor.shape) == 4:
        numpy_array = tensor.permute(0, 2, 3, 1).contiguous().squeeze().detach().cpu().numpy()
    else:
        numpy_array = tensor.contiguous().squeeze().detach().cpu().numpy()

    return numpy_array


def hm_plot(h):
    h = h / h.max()
    h = (h + 1) / 2
    plt.imshow(h.cpu(), vmin=0, vmax=1, cmap="seismic")
    plt.axis("off")
