import torch
import torch.nn.functional as F


def weighted_mse(y, y_pred, w=None):
    if w is None:
        return torch.mean((y - y_pred) * (y - y_pred))
    else:
        return torch.mean(w * (y - y_pred) * (y - y_pred))


def weighted_cross_entropy(y, y_pred, w=None, eps=1e-8):
    if w is None:
        res = -torch.mean(
            y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps)
        )
    else:
        res = -torch.mean(
            w * (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
        )
    return res


def LSIF_loss(y, y_pred):
    return torch.mean(1 / 2 * y * torch.square(y_pred) + (1 - y) * (-y_pred))


def dro_mse(y, y_pred, eta):
    bse = (y - y_pred) ** 2
    bse = F.relu(bse - eta)
    bse = bse**2
    return bse.mean()


def dro_ce(y, y_pred, eta, eps=1e-8):
    bce = -1 * y * torch.log(y_pred + eps) - (1 - y) * torch.log(1 - y_pred + eps)
    bce = F.relu(bce - eta)
    bce = bce**2
    return bce.mean()
