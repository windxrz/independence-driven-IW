import argparse

from stable_learning.utils.dataset import simulate_data


def parse_args(more_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10000)
    parser.add_argument("-p", type=int, default=10)
    parser.add_argument("--r-train", type=float, default=2.5)
    parser.add_argument("--mlp-scale", type=float, default=1)
    parser.add_argument("--clip", type=float, default=2)
    parser.add_argument("--dim-v", type=int, default=2)
    parser.add_argument("--category", type=str, choices=["mlp", "poly"], default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp-seed", type=int, default=2333)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--method-seed", type=int, default=0)

    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.001)

    if more_args is not None:
        more_args(parser)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data = simulate_data(
        args.n,
        args.p,
        args.r_train,
        args.mlp_scale,
        args.clip,
        args.dim_v,
        args.category,
        args.seed,
        args.mlp_seed,
    )
    print(data)


if __name__ == "__main__":
    main()
