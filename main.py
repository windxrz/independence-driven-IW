import argparse
import json
import os

import numpy as np
from termcolor import colored

from idiw.methods.methods import get_method
from idiw.utils.dataset import simulate_data
from idiw.utils.mlp import MLPRegressor


def parse_args(more_args=None):
    parser = argparse.ArgumentParser()
    # simulation data parameters
    parser.add_argument("-n", type=int, default=10000)
    parser.add_argument("-p", type=int, default=10)
    parser.add_argument("--r-train", type=float, default=2.5)
    parser.add_argument("--mlp-scale", type=float, default=1)
    parser.add_argument("--clip", type=float, default=2)
    parser.add_argument("--dim-v", type=int, default=2)
    parser.add_argument("--category", type=str, choices=["mlp", "poly"], default="poly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=2333)

    # methods
    subparser = parser.add_subparsers(dest="method")

    ols = subparser.add_parser("OLS")

    lasso = subparser.add_parser("LASSO")
    lasso.add_argument("--alpha", type=float, default=1.0)

    # others
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)

    # parser.add_argument("--nonlinear", action="store_true")
    # parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument("--epoch", type=int, default=1000)
    # parser.add_argument("--optim", type=str, default="Adam")
    # parser.add_argument("--lr", type=float, default=0.001)

    if more_args is not None:
        more_args(parser)

    args = parser.parse_args()
    return args


def get_feature_selection(data_name, data, method, method_seed):
    print(
        colored("Running for feature selection method {}".format(method.name), "blue")
    )
    ans_path = os.path.join("results", data_name, method.name)
    if not os.path.exists(ans_path):
        os.makedirs(ans_path)
    ans_filename = os.path.join(ans_path, "{}.json".format(method_seed))
    if os.path.exists(ans_filename):
        print(colored("Reading from previous runs", "blue"))
        with open(ans_filename, "r") as f:
            ans = json.loads(f.read())
            f.close()
    else:
        print(colored("Running", "blue"))
        train = list(data["train"].values())[0]
        X = train["X"]
        y = train["y"]
        feature_importance = method.feature_importance(X, y)
        idx = np.argsort(feature_importance)[::-1].tolist()

        ans = {
            "importance": feature_importance,
            "ranking": idx,
        }

        with open(ans_filename, "w") as f:
            f.write(json.dumps(ans, indent=4))
            f.close()

    return ans


def main():
    args = parse_args()
    print(args)

    data_name, data = simulate_data(args)
    method = get_method(args.method)(args)
    feature_res = get_feature_selection(data_name, data, method, args.method_seed)
    print(feature_res)


if __name__ == "__main__":
    main()
