import argparse
import random

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt


def manual_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(more_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=[
            "adult",
            # "compas",
            # "sim_simple",
            # "sim_kk_modified",
            # "sim_simple_5",
            # "sim_standard_kk",
            # "sim_my",
            "sim_2022_regression",
            # "sim_2022_classification",
            "sim_2022_regression_no_mlp",
            "house",
        ],
    )
    parser.add_argument("--drop", type=bool, default=True)
    parser.add_argument("-n", type=int, default=10000)
    parser.add_argument("-p", type=int, default=10)
    parser.add_argument("--r-train", type=float, default=2.5)
    parser.add_argument("--ratio", type=float, default=0.0)
    parser.add_argument("--dim-v", type=int, default=2)
    parser.add_argument("--category", type=str, default="mlp")
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument("--mlp-scale", type=float, default=1)
    parser.add_argument("--mlp-seed", type=int, default=2333)
    parser.add_argument("--clip", type=float, default=2)

    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--least-steps", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tensorboard", type=str2bool, default=False)
    parser.add_argument("--linear-pred", type=str2bool, default=True)

    if more_args is not None:
        more_args(parser)

    args = parser.parse_args()
    if args.dataset != "simulation":
        args.dag_type = None
    return args


def cross_entropy(y, y_pred, w=None, eps=1e-8):
    if w is None:
        res = -torch.sum(
            (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
        )
    else:
        res = -torch.sum(
            w * (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps))
        )
    return res


def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.T, x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.T)
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).T, x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.T)
    return res


def corr(x, w=None):
    if w is None:
        n = x.shape[0]
        w = torch.ones(n) / n
    w = w.view(-1, 1)
    covariance = cov(x, w)
    var = torch.sum(w * (x * x), dim=0).view(-1, 1) - torch.square(
        torch.sum(w * x, dim=0).view(-1, 1)
    )
    if torch.min(var) <= 0:
        print("error: ", torch.min(var))
    std = torch.sqrt(var)
    res = covariance / torch.matmul(std, std.T)
    return res


def plot_heatmap(mat, filename):
    plt.clf()
    sns.heatmap(mat.cpu(), center=0)
    plt.savefig(filename, bbox_inches="tight")


def plot_cov(x, filename, w=None):
    res = cov(x, w)
    plot_heatmap(res, filename)


def plot_corr(x, filename, w=None):
    res = corr(x, w)
    plot_heatmap(res, filename)
