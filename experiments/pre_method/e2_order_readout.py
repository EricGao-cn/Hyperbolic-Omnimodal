"""E2: do existing models encode the evidence order, or erase it?

For each model's feature dump, take its only radial quantity (geodesic radius
for hyperbolic models, raw projection norm for Euclidean ones; see
common.specificity_readout) and correlate it with the data-side measures:

- captions: readout of the caption vs E1a fan-out (a model that encodes the
  order puts generic captions, with large fan-out, at small radius: rho < 0);
- clips: readout of v (or a) vs log(1 + E1b accepted count under {v} (or {a}))
  (same expected sign);
- modality asymmetry: on clips where E1b says one modality is more ambiguous
  than the other, how often the readout agrees (smaller readout for the more
  ambiguous modality). Euclidean projection norms come from different heads,
  so this agreement is only meaningful for the hyperbolic models.

Dump dataset names must match the names given to E1a/E1b.

    python experiments/pre_method/e2_order_readout.py --out OUT/e2 --e1a OUT/e1a --e1b OUT/e1b \
        vast=OUT/features/vast pmrl=OUT/features/pmrl k_e0=OUT/features/k_e0
"""
import argparse
import json
import math
from pathlib import Path

import torch

from common import bootstrap_ci, list_datasets, load_dump, spearman, specificity_readout, write_json


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def aligned(ids, mapping, key=None):
    values = []
    for clip in ids:
        entry = mapping.get(clip) if mapping else None
        if entry is not None and key is not None:
            entry = entry.get(key)
        values.append(float("nan") if entry is None else float(entry))
    return torch.tensor(values, dtype=torch.float64)


def correlation(x, y, n_boot):
    mask = torch.isfinite(x) & torch.isfinite(y)
    if mask.sum() < 10:
        return {"rho": float("nan"), "n": int(mask.sum())}
    return {"rho": spearman(x[mask], y[mask]), "ci95": bootstrap_ci(spearman, x[mask], y[mask], n_boot),
            "n": int(mask.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--e1a", required=True, help="E1a output directory")
    parser.add_argument("--e1b", default=None, help="E1b output directory")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("dumps", nargs="+", help="name=feature_dump_dir")
    args = parser.parse_args()

    report = {}
    lines = ["| model | dataset | readout | ρ(caption, fan-out) | ρ(v, count_v) | ρ(a, count_a) | v/a agreement |",
             "|---|---|---|---:|---:|---:|---:|"]
    for spec in args.dumps:
        name, directory = spec.split("=", 1)
        report[name] = {}
        for dataset in list_datasets(directory):
            fanout = read_json(Path(args.e1a) / f"e1a_fanout_{dataset}.json")
            counts = read_json(Path(args.e1b) / f"e1b_counts_{dataset}.json") if args.e1b else None
            if fanout is None and counts is None:
                continue
            dump = load_dump(directory, dataset)
            ids = dump["ids"]
            entry = {}
            readouts = {m: specificity_readout(dump, m) for m in ("t", "v", "a") if m in dump["proj"]}
            entry["readout"] = {m: kind for m, (_, kind) in readouts.items()}
            if fanout is not None:
                entry["caption_vs_fanout"] = correlation(readouts["t"][0], aligned(ids, fanout), args.bootstrap)
            if counts is not None:
                for m in ("v", "a"):
                    entry[f"{m}_vs_count"] = correlation(
                        readouts[m][0], torch.log1p(aligned(ids, counts, m)), args.bootstrap)
                count_v, count_a = aligned(ids, counts, "v"), aligned(ids, counts, "a")
                differs = torch.isfinite(count_v) & torch.isfinite(count_a) & (count_v != count_a)
                # the more ambiguous modality (larger count) should have the smaller readout
                agree = torch.sign(readouts["v"][0] - readouts["a"][0]) == torch.sign(count_a - count_v)
                entry["v_a_agreement"] = {"rate": agree[differs].double().mean().item() if differs.any() else float("nan"),
                                          "n": int(differs.sum()),
                                          "comparable": readouts["v"][1] == "geodesic_radius"}
            report[name][dataset] = entry

            def fmt(key):
                value = entry.get(key, {}).get("rho", float("nan"))
                return "–" if math.isnan(value) else f"{value:+.2f}"

            agreement = entry.get("v_a_agreement", {}).get("rate", float("nan"))
            lines.append(f"| {name} | {dataset} | {entry['readout']['t']} | {fmt('caption_vs_fanout')} | "
                         f"{fmt('v_vs_count')} | {fmt('a_vs_count')} | "
                         f"{'–' if math.isnan(agreement) else f'{agreement:.2f}'} |")
    out = Path(args.out)
    write_json(out / "e2_order_readout.json", report)
    (out / "e2_order_readout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
