"""Shared I/O, geometry readouts and statistics for the pre-method diagnostics.

A feature dump is one torch file per dataset and rank, written by
extract_features.py (or synthetic.py):

    {'meta':   {model_type, dataset, checkpoint, curvature (or None), ...},
     'ids':    [clip id, ...],                 one entry per clip
     'captions': [first caption, ...],
     'pooled': {m: [N, d_m]},                  backbone features before projection
     'proj':   {m: [N, D]},                    raw contrastive projection (pre-normalization)
     'emb':    {m: [N, D]},                    the model's own retrieval embedding
                                               (unit vector, or Lorentz spatial coordinates)
     'radius_delta': {m: [N]},                 direction-radius lift only: log multiplier
     'truth':  {...}}                          synthetic data only

Modalities are keyed t (caption), v, a, s (subtitle, when present).
"""
import json
import math
from pathlib import Path

import torch

MODALITIES = ("t", "v", "a", "s")


# --------------------------------------------------------------------------- I/O

def shard_path(out_dir, dataset, rank):
    return Path(out_dir) / f"{dataset}.rank{rank}.pt"


def load_dump(out_dir, dataset):
    """Merge rank shards of one dataset; drop duplicate ids, keep first."""
    shards = sorted(Path(out_dir).glob(f"{dataset}.rank*.pt"))
    if not shards:
        single = Path(out_dir) / f"{dataset}.pt"
        if not single.exists():
            raise FileNotFoundError(f"no dump for {dataset} in {out_dir}")
        shards = [single]
    parts = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    merged = {"meta": parts[0]["meta"], "ids": [], "captions": []}
    tensor_groups = [k for k, v in parts[0].items() if isinstance(v, dict) and k != "meta"]
    keep = []
    seen = set()
    offset = 0
    for part in parts:
        for i, identifier in enumerate(part["ids"]):
            key = str(identifier)
            if key not in seen:
                seen.add(key)
                keep.append(offset + i)
                merged["ids"].append(key)
                merged["captions"].append(part["captions"][i])
        offset += len(part["ids"])
    index = torch.tensor(keep, dtype=torch.long)
    for group in tensor_groups:
        merged[group] = {}
        for key in parts[0][group]:
            values = [p[group][key] for p in parts]
            if isinstance(values[0], torch.Tensor):
                merged[group][key] = torch.cat(values)[index]
            else:
                flat = [x for v in values for x in v]
                merged[group][key] = [flat[i] for i in keep]
    return merged


def list_datasets(out_dir):
    names = {p.name.split(".rank")[0] for p in Path(out_dir).glob("*.rank*.pt")}
    names |= {p.stem for p in Path(out_dir).glob("*.pt") if ".rank" not in p.name}
    return sorted(names)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


# ---------------------------------------------------------------------- geometry

def lorentz_radius(spatial, curvature):
    """Geodesic distance to the origin from Lorentz spatial coordinates."""
    c = torch.as_tensor(curvature, dtype=torch.float64)
    return torch.asinh(c.sqrt() * spatial.double().norm(dim=-1)) / c.sqrt()


def lorentz_log0(spatial, curvature):
    radius = lorentz_radius(spatial, curvature)
    norm = spatial.double().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return spatial.double() / norm * radius.unsqueeze(-1)


def specificity_readout(dump, modality):
    """The model's only radial quantity for a modality.

    Hyperbolic models: geodesic radius of the embedding.
    Euclidean models: norm of the raw projection before L2 normalization.
    """
    curvature = dump["meta"].get("curvature")
    if curvature is not None and modality in dump.get("emb", {}) and dump["meta"].get("lifted", {}).get(modality, False):
        return lorentz_radius(dump["emb"][modality], curvature), "geodesic_radius"
    return dump["proj"][modality].double().norm(dim=-1), "projection_norm"


# -------------------------------------------------------------------- statistics

def rankdata(values):
    """Average ranks (ties share the mean rank), 0-based."""
    values = torch.as_tensor(values, dtype=torch.float64).detach().cpu().flatten()
    sorted_values, order = torch.sort(values)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0)
    mean_rank = (ends - 1 + ends - counts).double() / 2
    ranks = torch.empty_like(values)
    ranks[order] = torch.repeat_interleave(mean_rank, counts)
    return ranks


def pearson(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).detach().cpu().flatten()
    y = torch.as_tensor(y, dtype=torch.float64).detach().cpu().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if x.numel() < 3:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    denominator = (x.norm() * y.norm()).item()
    return float("nan") if denominator == 0 else (x @ y).item() / denominator


def spearman(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).detach().cpu().flatten()
    y = torch.as_tensor(y, dtype=torch.float64).detach().cpu().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    return pearson(rankdata(x[mask]), rankdata(y[mask]))


def bootstrap_ci(statistic, x, y, n_boot=1000, seed=0, alpha=0.05):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(n_boot):
        index = torch.randint(x.numel(), (x.numel(),), generator=generator)
        values.append(statistic(x[index], y[index]))
    values = torch.tensor([v for v in values if math.isfinite(v)], dtype=torch.float64)
    if values.numel() == 0:
        return [float("nan"), float("nan")]
    return [torch.quantile(values, alpha / 2).item(), torch.quantile(values, 1 - alpha / 2).item()]


def quantiles(values, qs=(0.1, 0.5, 0.9)):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {}
    out = {f"q{int(q * 100)}": torch.quantile(values, q).item() for q in qs}
    out["mean"] = values.mean().item()
    return out


def recall_at_k(scores, positives, ks=(1, 5, 10)):
    """scores [Q, C]; positives [Q] candidate index of the single positive."""
    positive_scores = scores.gather(1, positives[:, None])
    rank = (scores > positive_scores).sum(1)
    return {f"R@{k}": 100.0 * (rank < k).double().mean().item() for k in ks}
