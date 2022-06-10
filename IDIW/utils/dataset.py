import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
from IDIW.utils.mlp import MLP
from IDIW.utils.utils import manual_seed
from termcolor import colored
from torch import nn


def random_mlp(shapes, scale, mlp_seed=2333):
    manual_seed(mlp_seed)
    mlp = MLP(shapes)
    for parameter in mlp.parameters():
        nn.init.uniform_(parameter, -scale, scale)
    mlp.freeze()
    return mlp


def correlation_sample(s, v, y, r, p, n, dim_v):
    x = np.concatenate([s, v], axis=1)
    n0 = x.shape[0]
    prob = np.ones(n0)

    for idx in range(int(p / 10) * dim_v):
        d = np.abs(x[:, -idx - 1] - np.sign(r) * y)
        prob = prob * np.power(np.abs(r), -10 * d)
    prob = prob / np.sum(prob)
    idx = np.random.choice(range(n0), n, p=prob)
    x = x[idx, :]
    y = y[idx]
    return x, y


def generate_sim_regression(
    n,
    p,
    rs,
    clip,
    dim_v,
    category,
    mlp,
):

    p_s = int(p * 0.5)
    p_v = p - p_s

    n0 = n * 100

    data = {}
    for r in rs:
        z = np.random.normal(0, 1, (n0, p_s + 1))
        s = 0.8 * z[:, :p_s] + 0.2 * z[:, 1:]
        v = np.random.normal(0, 1, (n0, p_v))
        s = np.clip(s, -clip, clip)
        v = np.clip(v, -clip, clip)

        beta = []
        for _ in range(p_s):
            beta.extend([1 / 3, -2 / 3, 1, -1 / 3, 2 / 3, -1])
        beta = beta[:p_s]

        tensor = torch.from_numpy(s[:, :3]).float()
        if category == "mlp":
            y_nonlinear = mlp(tensor).view(-1).detach().cpu().numpy()
            y = s @ beta + y_nonlinear
        elif category == "poly":
            y = s @ beta + s[:, 0] * s[:, 1] * s[:, 2] / 4

        x, y = correlation_sample(s, v, y, r, p, n, dim_v)
        y = y + np.random.normal(0, 0.3, size=y.shape[0])
        y = y.reshape(-1, 1)
        data[r] = {"X": x, "y": y}

    return data


def simulate_data(args):
    n = args.n
    p = args.p
    r_train = args.r_train
    mlp_scale = args.mlp_scale
    clip = args.clip
    dim_v = args.dim_v
    category = args.category
    seed = args.seed
    mlp_seed = args.mlp_seed
    name = "sim_n_{}_p_{}_r_train_{}_mlp_{}_clip_{}_dimv_{}_category_{}{}".format(
        n,
        p,
        r_train,
        mlp_scale,
        clip,
        dim_v,
        category,
        "_mlpseed_{}".format(mlp_seed) if category == "mlp" else "",
    )
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/{}.pkl".format(name)):
        print(colored("Generating dataset, {}".format(name), "blue"))
        r_test = [-3.0, -2.5, -2.0, -1.5, -1.3, 1.3, 1.5, 2.0, 2.5, 3.0]
        if category == "mlp":
            mlp = random_mlp([3, 3, 3, 1], scale=mlp_scale, mlp_seed=mlp_seed)
        else:
            mlp = None

        manual_seed(seed)
        data_train = generate_sim_regression(
            n, p, [r_train], clip, dim_v, category, mlp
        )
        data_test = generate_sim_regression(n, p, r_test, clip, dim_v, category, mlp)
        data = {
            "train": data_train,
            "test": data_test,
        }
        with open("data/{}.pkl".format(name), "wb") as f:
            f.write(pkl.dumps(data))
            f.close()
    else:
        print(colored("Reading from data, {}".format(name), "blue"))
        with open("data/{}.pkl".format(name), "rb") as f:
            data = pkl.loads(f.read())
            f.close()

    return name, data
