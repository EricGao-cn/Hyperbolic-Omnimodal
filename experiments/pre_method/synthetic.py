"""E5: synthetic data with a known evidence order, in the feature-dump format.

Each instance is a leaf of a complete tree (branching b, depth L). Each
modality reveals a random subset of the levels of the leaf's root path: text
mostly shallow levels (generic), video many levels, audio a few. A subset of
modalities therefore reveals the union of their levels, and the number of
leaves consistent with it is b^(L - |union|): the order the pilot should
recover. Every modality also carries private, instance-specific noise that no
other modality shares.

Features are produced by a fixed random two-layer network per modality from
the concatenated level embeddings (unrevealed levels contribute zeros), so the
pilot cannot read the depth off a coordinate.

    python experiments/pre_method/synthetic.py --out OUT/synthetic
    python experiments/pre_method/train_pilot.py --features OUT/synthetic --train syn_train \
        --holdout syn_holdout --test syn_test --geometry lorentz --objective order --out ...

Writes syn_train.pt, syn_holdout.pt and syn_test.pt; each carries truth_levels.
"""
import argparse
from pathlib import Path

import torch

REVEAL = {"t": [0.95, 0.8, 0.4, 0.15, 0.05, 0.05], "v": [0.9, 0.85, 0.8, 0.7, 0.6, 0.5], "a": [0.7, 0.5, 0.35, 0.25, 0.15, 0.1]}


def make(n, branching, depth, private_dim, noise, generator, tables, nets):
    leaves = torch.randint(branching ** depth, (n,), generator=generator)
    digits = torch.stack([(leaves // branching ** (depth - 1 - level)) % branching for level in range(depth)], 1)
    prefix = torch.zeros(n, depth, dtype=torch.long)
    for level in range(depth):
        powers = torch.tensor([branching ** (level - j) for j in range(level + 1)])
        prefix[:, level] = (digits[:, :level + 1] * powers).sum(1)
    dump = {"ids": [f"syn{i}" for i in range(n)], "captions": [""] * n, "pooled": {}, "proj": {}, "emb": {},
            "truth_levels": {}, "truth_leaf": {"leaf": leaves}}
    for m, rates in REVEAL.items():
        p = torch.tensor(rates[:depth])
        levels = torch.rand(n, depth, generator=generator) < p
        levels[levels.sum(1) == 0, 0] = True                          # every modality reveals something
        parts = [tables[level][prefix[:, level]] * levels[:, level:level + 1] for level in range(depth)]
        private = torch.randn(n, private_dim, generator=generator)
        hidden = torch.cat(parts + [private], 1)
        w1, w2 = nets[m]
        x = torch.tanh(hidden @ w1) @ w2
        x = x + noise * torch.randn(x.shape, generator=generator)
        dump["pooled"][m] = x.float()
        dump["truth_levels"][m] = levels
    return dump


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--level-dim", type=int, default=16)
    parser.add_argument("--private-dim", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-holdout", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    if args.depth > len(REVEAL["t"]):
        raise SystemExit(f"--depth at most {len(REVEAL['t'])}")

    g = torch.Generator().manual_seed(args.seed)
    tables = [torch.randn(args.branching ** (level + 1), args.level_dim, generator=g) for level in range(args.depth)]
    hidden_dim = args.depth * args.level_dim + args.private_dim
    nets = {m: (torch.randn(hidden_dim, 4 * args.feature_dim, generator=g) / hidden_dim ** 0.5,
                torch.randn(4 * args.feature_dim, args.feature_dim, generator=g) / (4 * args.feature_dim) ** 0.5)
            for m in REVEAL}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, n in (("syn_train", args.n_train), ("syn_holdout", args.n_holdout), ("syn_test", args.n_test)):
        dump = make(n, args.branching, args.depth, args.private_dim, args.noise, g, tables, nets)
        dump["meta"] = dict(model_type="synthetic", dataset=name, curvature=None, lifted={}, **vars(args))
        torch.save(dump, out / f"{name}.pt")
        print(f"{name}: {n} instances -> {out / f'{name}.pt'}")


if __name__ == "__main__":
    main()
