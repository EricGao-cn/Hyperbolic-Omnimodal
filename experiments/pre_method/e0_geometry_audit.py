"""E0: did the earlier hyperbolic checkpoints use curvature at all?

Reads feature dumps from extract_features.py (one directory per checkpoint)
and reports, per checkpoint, dataset and modality:
- learned curvature c and the modality log-scales (and whether they sit at the
  ceiling of their clamp, log-scale <= 0);
- sqrt(c) * r, the scale-free radius (the space is locally flat when it is << 1);
- how often the direction-radius multiplier is saturated at its (1/3, 3) bound;
- distortion d_H / ||log_0 x - log_0 y|| on matched caption-clip pairs and on
  each caption's nearest non-matching clip (1 means the model used the space as
  if it were flat; hyperbolic distance is never smaller than this ratio's
  denominator).

    python experiments/pre_method/e0_geometry_audit.py --out OUT/e0 \
        k_e0=OUT/features/k_e0 et=OUT/features/et m_spectral=OUT/features/m_spectral
"""
import argparse
import math
from pathlib import Path

import torch

from common import list_datasets, load_dump, lorentz_log0, lorentz_radius, quantiles, write_json

SATURATION = 0.95 * math.log(3.0)
FLAT_SCALED_RADIUS = 1.0
FLAT_DISTORTION = 1.1


def matched_geodesic(x, y, c):
    c = torch.as_tensor(c, dtype=torch.float64)
    tx = (1 / c + x.square().sum(-1)).sqrt()
    ty = (1 / c + y.square().sum(-1)).sqrt()
    inner = (c * (tx * ty - (x * y).sum(-1))).clamp_min(1.0)
    return 2 * torch.asinh(((inner - 1) / 2).clamp_min(0).sqrt()) / c.sqrt()


def all_geodesic(x, y, c, chunk=1024):
    c = torch.as_tensor(c, dtype=torch.float64)
    ty = (1 / c + y.square().sum(-1)).sqrt()
    rows = []
    for xs in x.split(chunk):
        tx = (1 / c + xs.square().sum(-1)).sqrt()
        inner = (c * (tx[:, None] * ty[None] - xs @ y.T)).clamp_min(1.0)
        rows.append(2 * torch.asinh(((inner - 1) / 2).clamp_min(0).sqrt()) / c.sqrt())
    return torch.cat(rows)


def audit_dataset(dump):
    meta = dump["meta"]
    c = meta["curvature"]
    result = {"clips": len(dump["ids"])}
    emb = {m: x.double() for m, x in dump["emb"].items() if meta["lifted"].get(m)}
    for m, x in emb.items():
        radius = lorentz_radius(x, c)
        scaled = math.sqrt(c) * radius
        result[f"scaled_radius_{m}"] = quantiles(scaled)
        result[f"radius_{m}"] = quantiles(radius)
        result[f"fraction_scaled_radius_below_1_{m}"] = (scaled < 1).double().mean().item()
    for m, delta in dump.get("radius_delta", {}).items():
        result[f"multiplier_saturated_fraction_{m}"] = (delta.abs() > SATURATION).double().mean().item()
    if "t" in emb:
        log_t = lorentz_log0(emb["t"], c)
        for m in ("v", "a"):
            if m not in emb:
                continue
            log_m = lorentz_log0(emb[m], c)
            positive = matched_geodesic(emb["t"], emb[m], c)
            positive_flat = (log_t - log_m).norm(dim=-1)
            result[f"distortion_positive_t{m}"] = quantiles(positive / positive_flat.clamp_min(1e-12))
            distances = all_geodesic(emb["t"], emb[m], c)
            distances.fill_diagonal_(float("inf"))
            nearest = distances.argmin(dim=1)
            negative = distances.gather(1, nearest[:, None]).squeeze(1)
            negative_flat = (log_t - log_m[nearest]).norm(dim=-1)
            result[f"distortion_hard_negative_t{m}"] = quantiles(negative / negative_flat.clamp_min(1e-12))
    medians_radius = [v["q50"] for k, v in result.items() if k.startswith("scaled_radius_")]
    medians_ratio = [v["q50"] for k, v in result.items() if k.startswith("distortion_")]
    result["near_flat"] = bool(medians_radius and medians_ratio
                               and max(medians_radius) < FLAT_SCALED_RADIUS
                               and max(medians_ratio) < FLAT_DISTORTION)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("dumps", nargs="+", help="name=feature_dump_dir")
    args = parser.parse_args()

    report = {}
    lines = ["| checkpoint | dataset | c | median sqrt(c)·r (t/v/a) | median distortion pos / hard-neg (tv) | multiplier saturated (t/v/a) | near flat |",
             "|---|---|---:|---|---|---|---|"]
    for spec in args.dumps:
        name, directory = spec.split("=", 1)
        report[name] = {}
        for dataset in list_datasets(directory):
            dump = load_dump(directory, dataset)
            meta = dump["meta"]
            if meta.get("curvature") is None:
                report[name][dataset] = {"skipped": f"{meta['model_type']} has no Lorentz embedding"}
                continue
            entry = audit_dataset(dump)
            entry["curvature"] = meta["curvature"]
            entry["log_modality_scales"] = meta["log_modality_scales"]
            entry["scale_at_ceiling"] = {m: s >= -1e-6 for m, s in meta["log_modality_scales"].items()}
            report[name][dataset] = entry

            def median(key):
                value = entry.get(key)
                return f"{value['q50']:.3f}" if value else "–"

            saturated = "/".join(f"{entry.get(f'multiplier_saturated_fraction_{m}', float('nan')):.2f}" for m in "tva")
            lines.append(f"| {name} | {dataset} | {meta['curvature']:.3f} | "
                         f"{'/'.join(median(f'scaled_radius_{m}') for m in 'tva')} | "
                         f"{median('distortion_positive_tv')} / {median('distortion_hard_negative_tv')} | "
                         f"{saturated} | {entry['near_flat']} |")
    out = Path(args.out)
    write_json(out / "e0_geometry_audit.json", report)
    (out / "e0_geometry_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
