import numpy as np
from scipy import ndimage


def calc_confusion_matrix(gt, pred, tolerance=0):
    """
    :param gt: Groundtruth mask.
    :param pred: Prediction mask.
    :param tolerance: Tolerance of pixel offset.
    :return: (precision_TP, precision_TPFP, recall_TP, recall_TPFN)
    """
    dilated_gt = gt
    dilated_pred = pred
    if tolerance > 0:
        structure = np.ones((1 + 2 * tolerance, 1 + 2 * tolerance))
        dilated_gt = ndimage.binary_dilation(gt, structure=structure)
        dilated_pred = ndimage.binary_dilation(pred, structure=structure)

    prec_TP = (pred & dilated_gt).sum()
    prec_TPFP = pred.sum()
    reca_TP = (gt & dilated_pred).sum()
    reca_TPFN = gt.sum()

    return prec_TP, prec_TPFP, reca_TP, reca_TPFN
