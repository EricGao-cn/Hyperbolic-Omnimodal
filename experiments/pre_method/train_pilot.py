"""E3/E4: train one head-only pilot cell on cached features and evaluate it.

Inputs are feature dumps from extract_features.py (pooled backbone features of
the VAST checkpoint) or from synthetic.py. Selection uses the holdout dump
only; test dumps are evaluated once, with the selected state.

    python experiments/pre_method/train_pilot.py --features OUT/features/vast \
        --train finetune_area_train --holdout vast_holdout \
        --test msrvtt_ret vatex_ret didemo_ret activitynet_ret \
        --e1a OUT/e1a --e1b OUT/e1b \
        --geometry lorentz --objective order --width 64 --out OUT/pilot/lorentz_order_w64

See README.md for the grid of cells.
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from common import load_dump, write_json  # noqa: E402
from pilot.evaluate import evaluate  # noqa: E402
from pilot.geometry import build_space  # noqa: E402
from pilot.model import SubsetEncoder  # noqa: E402
from pilot.objectives import OBJECTIVES, Objective  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="feature dump directory")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--holdout", required=True)
    p.add_argument("--test", nargs="*", default=[])
    p.add_argument("--e1a", default=None, help="E1a output dir (caption fan-out per test set)")
    p.add_argument("--e1b", default=None, help="E1b output dir (ITM counts per test set)")
    p.add_argument("--out", required=True)
    p.add_argument("--modalities", default="tva")
    p.add_argument("--geometry", choices=("euclid", "lorentz"), required=True)
    p.add_argument("--objective", choices=OBJECTIVES, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--encoder", choices=("fusion", "centroid", "perron"), default="fusion")
    p.add_argument("--order", choices=("join", "meet"), default="join")
    p.add_argument("--text-mode", choices=("lattice", "query"), default="lattice")
    p.add_argument("--score", choices=("dist", "gromov", "cos"), default="dist")
    p.add_argument("--order-weight", type=float, default=0.1)
    p.add_argument("--cone-k", type=float, default=0.1)
    p.add_argument("--pairs-per-step", type=int, default=0, help="0 = all subset pairs every step")
    p.add_argument("--curvature-init", type=float, default=1.0)
    p.add_argument("--fixed-curvature", action="store_true")
    p.add_argument("--tau-init", type=float, default=0.07)
    p.add_argument("--fixed-tau", action="store_true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", action="store_true", help="also save the selected state dict")
    return p.parse_args(argv)


def load_features(directory, names, modalities, max_items=0, seed=50):
    dumps = [load_dump(directory, n) for n in names]
    data = {"ids": [i for d in dumps for i in d["ids"]],
            "pooled": {m: torch.cat([d["pooled"][m] for d in dumps]) for m in modalities}}
    if all("truth_levels" in d for d in dumps):
        data["truth_levels"] = {m: torch.cat([d["truth_levels"][m] for d in dumps]) for m in modalities}
    if max_items and len(data["ids"]) > max_items:
        keep = torch.tensor(sorted(random.Random(seed).sample(range(len(data["ids"])), max_items)))
        data["ids"] = [data["ids"][i] for i in keep.tolist()]
        data["pooled"] = {m: x[keep] for m, x in data["pooled"].items()}
        if "truth_levels" in data:
            data["truth_levels"] = {m: x[keep] for m, x in data["truth_levels"].items()}
    return data


def read_optional(directory, pattern, name):
    if not directory:
        return None
    path = Path(directory) / pattern.format(name)
    return json.loads(path.read_text()) if path.exists() else None


def lr_at(step, total, warmup):
    warm = max(1, int(warmup * total))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    modalities = tuple(args.modalities)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train = load_features(args.features, args.train, modalities, args.max_train, args.seed)
    holdout = load_features(args.features, [args.holdout], modalities)
    input_dims = {m: x.shape[1] for m, x in train["pooled"].items()}
    train_x = {m: x.to(device) for m, x in train["pooled"].items()}
    n_train = len(train["ids"])

    space = build_space(args.geometry, curvature_init=args.curvature_init,
                        learn_curvature=not args.fixed_curvature, cone_k=args.cone_k)
    model = SubsetEncoder(input_dims, modalities, space, args.width, args.encoder).to(device)
    log_logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / args.tau_init), device=device),
                                         requires_grad=not args.fixed_tau)
    objective = Objective(args.objective, modalities, args.score, args.order, args.text_mode,
                          args.order_weight, args.pairs_per_step, seed=args.seed)
    params = [p for p in model.parameters() if p.requires_grad] + ([log_logit_scale] if not args.fixed_tau else [])
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, args.steps, args.warmup))

    write_json(out / "config.json", vars(args) | {"n_train": n_train, "input_dims": input_dims})
    log = (out / "log.jsonl").open("w")
    best = {"score": -1.0, "step": -1, "state": None, "holdout": None}
    generator = torch.Generator().manual_seed(args.seed)
    start = time.time()
    for step in range(args.steps):
        index = torch.randint(n_train, (args.batch,), generator=generator).to(device)
        batch = {m: x[index] for m, x in train_x.items()}
        logit_scale = log_logit_scale.clamp(max=math.log(100.0)).exp()
        loss, logs = objective(model, batch, logit_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        last = step + 1 == args.steps
        if (step + 1) % args.eval_every == 0 or last:
            metrics = evaluate(model, objective, holdout)
            record = dict(step=step + 1, loss=loss.item(), logit_scale=logit_scale.item(), elapsed=time.time() - start,
                          **logs, holdout_selection=metrics["selection_score"],
                          holdout_monotone_rate=metrics["monotone_rate"],
                          holdout_cone_join=metrics["cone_satisfied_join"],
                          curvature=metrics.get("curvature"))
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if metrics["selection_score"] > best["score"]:
                best = {"score": metrics["selection_score"], "step": step + 1, "holdout": metrics,
                        "state": copy.deepcopy({"model": model.state_dict(), "logit": log_logit_scale.detach()})}
    log.close()

    model.load_state_dict(best["state"]["model"])
    result = {"config": vars(args), "best_step": best["step"], "holdout": best["holdout"], "test": {}}
    for name in args.test:
        data = load_features(args.features, [name], modalities)
        fanout = read_optional(args.e1a, "e1a_fanout_{}.json", name)
        counts = read_optional(args.e1b, "e1b_counts_{}.json", name)
        result["test"][name] = evaluate(model, objective, data, fanout, counts)
    write_json(out / "metrics.json", result)
    if args.save:
        torch.save(best["state"], out / "best.pt")
    print(f"selected step {best['step']} (holdout {best['score']:.2f}); wrote {out / 'metrics.json'}")
    return result


if __name__ == "__main__":
    main()
