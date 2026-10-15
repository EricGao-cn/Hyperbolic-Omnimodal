"""Collect pilot metrics.json files into one table (E3/E4 decision readout).

    python experiments/pre_method/summarize_pilot.py OUT/pilot/*/ --out OUT/pilot/summary.md

Columns are test-set means over the given test datasets: Stage-1 R@1 for
T->v, T->a, T->va; the missing-modality drop T->va minus T->v; the order
metrics; and the E1a/E1b/truth correlations when present.
"""
import argparse
import json
import math
from pathlib import Path


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def row(path):
    metrics = json.loads((Path(path) / "metrics.json").read_text())
    config, tests = metrics["config"], metrics["test"].values()

    def r1(key):
        return mean([t["retrieval"].get(key, {}).get("R@1") for t in tests])

    def metric(key):
        return mean([t.get(key) for t in tests])

    return {
        "cell": Path(path).name, "geometry": config["geometry"], "objective": config["objective"],
        "width": config["width"], "encoder": config["encoder"], "order": config["order"],
        "text": config["text_mode"], "score": config["score"], "best_step": metrics["best_step"],
        "holdout": metrics["holdout"]["selection_score"],
        "t2v": r1("t2v"), "t2a": r1("t2a"), "t2va": r1("t2va"),
        "monotone": metric("monotone_rate"), "rho_size": metric("spearman_radius_vs_size"),
        "cone_join": metric("cone_satisfied_join"),
        "rho_fanout": metric("spearman_text_radius_vs_fanout"),
        "rho_audio_gain": metric("spearman_audio_gain_vs_itm"),
        "rho_true_depth": metric("spearman_radius_vs_true_depth"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = [row(c) for c in args.cells if (Path(c) / "metrics.json").exists()]
    if not rows:
        raise SystemExit("no metrics.json found")
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in sorted(rows, key=lambda r: (r["objective"], r["width"], r["geometry"])):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
