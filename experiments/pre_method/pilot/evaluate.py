"""Pilot evaluation: retrieval, missing-modality retrieval, order and attribution.

All metrics are Stage-1 (embedding scores only, no ITM).

- retrieval  T -> C and C -> T for every clip subset C (so T->{v} vs T->{v,a}
             is the missing-modality comparison), plus v <-> a.
- order      mean radius by subset size; Spearman(radius, |S|); the rate at
             which r(h(S)) < r(h(S')) on covering pairs S c S'; join-direction
             cone satisfaction on the same pairs.
- E1a        Spearman(r(h({t})), caption fan-out)            expected < 0
- E1b        Spearman(r(h({v,a})) - r(h({v})), log((1+n_v)/(1+n_va)))   expected > 0,
             and the same for the audio gain against n_a.
- truth      synthetic dumps (truth_levels): Spearman(r(h(S)), revealed depth of S)     expected > 0
"""
import math

import torch

from common import recall_at_k, spearman

from pilot.model import all_subsets


@torch.no_grad()
def embed_dataset(model, features, subsets, batch_size=1024):
    n = next(iter(features.values())).shape[0]
    out = {s: [] for s in subsets}
    for start in range(0, n, batch_size):
        chunk = {m: x[start:start + batch_size] for m, x in features.items()}
        for s in subsets:
            out[s].append(model(chunk, s).double())
    return {s: torch.cat(v) for s, v in out.items()}


def name(subset):
    return "".join(subset)


@torch.no_grad()
def evaluate(model, objective, data, fanout=None, counts=None):
    model.eval()
    device = next(model.parameters()).device
    features = {m: data["pooled"][m].to(device) for m in objective.modalities}
    subsets = all_subsets(objective.modalities)
    points = embed_dataset(model, features, subsets)
    n = points[subsets[0]].shape[0]
    positives = torch.arange(n, device=device)
    result = {"clips": n, "retrieval": {}}

    clip_subsets = [s for s in subsets if "t" not in s]
    for c in clip_subsets:
        scores = objective.retrieval_scores(model, points, ("t",), c)
        result["retrieval"][f"t2{name(c)}"] = recall_at_k(scores, positives)
        result["retrieval"][f"{name(c)}2t"] = recall_at_k(scores.T.contiguous(), positives)
    if ("v",) in points and ("a",) in points:
        scores = objective.retrieval_scores(model, points, ("v",), ("a",))
        result["retrieval"]["v2a"] = recall_at_k(scores, positives)
        result["retrieval"]["a2v"] = recall_at_k(scores.T.contiguous(), positives)
    full_clip = tuple(m for m in objective.modalities if m != "t")
    selection = [result["retrieval"][f"t2{name(s)}"]["R@1"] for s in clip_subsets if len(s) == 1]
    selection.append(result["retrieval"][f"t2{name(full_clip)}"]["R@1"])
    result["selection_score"] = sum(selection) / len(selection)

    space = model.space
    radius = {s: space.radius(p).double() for s, p in points.items()}
    result["radius_by_size"] = {
        str(k): torch.cat([radius[s] for s in subsets if len(s) == k]).mean().item()
        for k in range(1, len(objective.modalities) + 1)}
    sizes = torch.cat([torch.full((n,), float(len(s)), device=device) for s in subsets])
    result["spearman_radius_vs_size"] = spearman(torch.cat([radius[s] for s in subsets]).cpu(), sizes.cpu())
    covering = [(a, b) for a in subsets for b in subsets if len(b) == len(a) + 1 and set(a) < set(b)]
    result["monotone_rate"] = torch.stack([(radius[a] < radius[b]).double().mean() for a, b in covering]).mean().item()
    result["cone_satisfied_join"] = torch.stack(
        [(space.cone_violation(points[a], points[b]) <= 0).double().mean() for a, b in covering]).mean().item()
    result["cone_satisfied_meet"] = torch.stack(
        [(space.cone_violation(points[b], points[a]) <= 0).double().mean() for a, b in covering]).mean().item()
    if space.name == "lorentz":
        c = space.c.item()
        result["curvature"] = c
        result["scaled_radius_median"] = {name(s): (math.sqrt(c) * radius[s]).median().item() for s in subsets
                                          if len(s) in (1, len(objective.modalities))}

    ids = data["ids"]
    r = {s: radius[s].cpu() for s in subsets}
    if fanout is not None:
        target = torch.tensor([fanout.get(i, float("nan")) for i in ids], dtype=torch.float64)
        result["spearman_text_radius_vs_fanout"] = spearman(r[("t",)], target)
    if counts is not None and ("v", "a") in r:
        def column(key):
            return torch.tensor([counts.get(i, {}).get(key, float("nan")) for i in ids], dtype=torch.float64)
        n_v, n_a, n_va = column("v"), column("a"), column("va")
        result["spearman_audio_gain_vs_itm"] = spearman(r[("v", "a")] - r[("v",)], torch.log((1 + n_v) / (1 + n_va)))
        result["spearman_video_gain_vs_itm"] = spearman(r[("v", "a")] - r[("a",)], torch.log((1 + n_a) / (1 + n_va)))
    if data.get("truth_levels"):
        levels = {m: data["truth_levels"][m].bool() for m in objective.modalities}
        depth = {s: torch.stack([levels[m] for m in s]).any(0).sum(-1).double() for s in subsets}
        result["spearman_radius_vs_true_depth"] = spearman(torch.cat([r[s] for s in subsets]),
                                                           torch.cat([depth[s] for s in subsets]))
        result["spearman_radius_vs_true_depth_singletons"] = {
            m: spearman(r[(m,)], depth[(m,)]) for m in objective.modalities}
        if ("v", "a") in r:
            result["spearman_audio_gain_vs_true_gain"] = spearman(r[("v", "a")] - r[("v",)],
                                                                  depth[("v", "a")] - depth[("v",)])
    model.train()
    return result
