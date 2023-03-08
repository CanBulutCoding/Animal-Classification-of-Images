import numpy as np


def nll_loss(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    pos_pred_y = np.where(pred_y != 0, pred_y, 1e-3)
    neg_pred_y = np.where(1 - pred_y != 0, 1 - pred_y, 1e-3)

    t1 = true_y * np.log(pos_pred_y)

    t2 = (1 - true_y) * np.log(neg_pred_y)

    nll = -1 * (t1 + t2)
    return np.sum(nll)


def nll_loss_derivative(true_y: np.ndarray, pred_y: np.ndarray) -> float:
    pos_pred_y = np.where(pred_y != 0, pred_y, 1e-3)
    neg_pred_y = np.where(1 - pred_y != 0, 1 - pred_y, 1e-3)

    t1 = true_y / pos_pred_y
    t2 = (1 - true_y) / neg_pred_y

    dl = -1 * (t1 + t2)
    return dl
