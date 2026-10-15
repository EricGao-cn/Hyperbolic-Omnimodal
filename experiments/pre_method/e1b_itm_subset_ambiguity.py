"""E1b: does exposing more modalities narrow what a caption can match?

Uses the cross-attention ITM head of a trained checkpoint (VAST or PMRL) as a
near-oracle matcher. For each caption, a fixed pool of K candidate clips is
taken from Stage-1 scores (the positive is always included), and every
candidate is scored by the ITM head with its condition tokens restricted to a
modality subset: {v}, {a}, {v,a} and, when subtitles exist, {v,a,s}. The pool
is identical across subsets, so the accepted counts are comparable.

Prediction from conditioning: count(v,a) <= min(count(v), count(a)), and
count(a) >> count(v) for most captions. If the ITM head does not accept the
positive under single-modality conditions as often as under {v,a}, the measure
is unreliable for that subset; the summary flags this.

Single process only (the pool and the condition tokens stay in host memory):
    torchrun --nproc_per_node 1 experiments/pre_method/e1b_itm_subset_ambiguity.py \
        --out OUT/e1b --config CFG.json --pretrain_dir VAST_DIR

Writes e1b_<dataset>.pt (pools and probabilities), e1b_counts_<dataset>.json
({clip_id: {subset: accepted count}}, keyed by the caption's own clip),
e1b_summary.json and e1b_summary.md.
"""
import argparse
import time
from pathlib import Path

from runtime import build, build_loader, dataset_configs, first_captions, parse_own_args

import torch
import torch.distributed as dist
from easydict import EasyDict as edict

from common import quantiles, write_json


@torch.no_grad()
def collect(model, loader, with_subtitles):
    store = {"ids": [], "feat_t": [], "feat_v": [], "feat_a": [], "input_ids": [], "attention_mask": [],
             "cond_v": [], "cond_a": [], "cond_s": []}
    for batch in loader:
        batch = edict(batch)
        batch.raw_captions = first_captions(batch.raw_captions)
        store["ids"] += [str(i) for i in batch.ids]
        for m in ("t", "v", "a"):
            store[f"feat_{m}"].append(model.batch_get(batch, f"feat_{m}").double().cpu())
        tokens = model.batch_get(batch, "caption_tokens")
        store["input_ids"].append(tokens.input_ids.cpu())
        store["attention_mask"].append(tokens.attention_mask.cpu())
        store["cond_v"].append(model.batch_get(batch, "condition_feats_v").half().cpu())
        store["cond_a"].append(model.batch_get(batch, "condition_feats_a").half().cpu())
        if with_subtitles and getattr(batch, "raw_subtitles", None) is not None:
            store["cond_s"].append(model.batch_get(batch, "condition_feats_s").half().cpu())
    out = {k: (torch.cat(v) if v and isinstance(v[0], torch.Tensor) else v) for k, v in store.items()}
    if not store["cond_s"]:
        out.pop("cond_s")
    return out


def stage1_scores(model, data):
    t, v, a = data["feat_t"], data["feat_v"], data["feat_a"]
    if getattr(model, "hyperbolic_curvature", None) is not None:
        device = next(model.parameters()).device
        sim = lambda x, y: model.retrieval_similarity(x.to(device), y.to(device)).float().cpu()  # noqa: E731
    else:
        sim = lambda x, y: x @ y.T  # noqa: E731
    return sim(t, v) + sim(t, a)


@torch.no_grad()
def itm_pool(model, data, pool, subsets, batch_size):
    device = next(model.parameters()).device
    n, k = pool.shape
    probs = torch.zeros(n, k, len(subsets), dtype=torch.float16)
    for i in range(n):
        ids = data["input_ids"][i:i + 1].expand(k, -1).to(device)
        mask = data["attention_mask"][i:i + 1].expand(k, -1).to(device)
        for s, subset in enumerate(subsets):
            cond = torch.cat([data[f"cond_{m}"][pool[i]] for m in subset], dim=1)
            scores = []
            for start in range(0, k, batch_size):
                chunk = cond[start:start + batch_size].to(device)
                scores.append(model.compute_slice_scores(
                    chunk.to(next(model.multimodal_encoder.parameters()).dtype),
                    ids[start:start + batch_size], mask[start:start + batch_size]).float().cpu())
            probs[i, :, s] = torch.cat(scores).half()
        if i % 200 == 0:
            print(f"  ITM {i}/{n}", flush=True)
    return probs


def summarize(probs, subsets, threshold):
    names = ["".join(s) for s in subsets]
    accepted = probs.float() > threshold
    counts = accepted.sum(1)                                   # [n, subsets]
    positive = probs[:, 0].float()                             # pool column 0 is the positive
    ranks = (probs.float() > positive[:, None]).sum(1)         # 0 = positive ranked first
    summary = {"captions": probs.shape[0], "pool": probs.shape[1], "threshold": threshold, "subsets": {}}
    for s, name in enumerate(names):
        summary["subsets"][name] = {
            "positive_accept_rate": (positive[:, s] > threshold).double().mean().item(),
            "positive_rank1_rate": (ranks[:, s] == 0).double().mean().item(),
            "accepted_count": quantiles(counts[:, s].double(), (0.5, 0.9)),
            "fraction_count_le_1": (counts[:, s] <= 1).double().mean().item(),
        }
    index = {name: i for i, name in enumerate(names)}
    checks = {}
    if {"v", "a", "va"} <= index.keys():
        v, a, va = counts[:, index["v"]], counts[:, index["a"]], counts[:, index["va"]]
        checks["fraction_va_le_min_v_a"] = (va <= torch.minimum(v, a)).double().mean().item()
        checks["fraction_a_ge_v"] = (a >= v).double().mean().item()
        checks["median_log_ratio_v_over_va"] = torch.log((1 + v.double()) / (1 + va.double())).median().item()
        checks["median_log_ratio_a_over_va"] = torch.log((1 + a.double()) / (1 + va.double())).median().item()
        reference = summary["subsets"]["va"]["positive_accept_rate"]
        checks["single_modality_itm_reliable"] = {
            m: summary["subsets"][m]["positive_accept_rate"] >= 0.5 * reference for m in ("v", "a")}
    if {"va", "vas"} <= index.keys():
        checks["fraction_vas_le_va"] = (counts[:, index["vas"]] <= counts[:, index["va"]]).double().mean().item()
    summary["checks"] = checks
    return summary, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pool", type=int, default=50, help="candidates per caption")
    parser.add_argument("--threshold", type=float, default=0.5, help="ITM match probability")
    parser.add_argument("--itm-batch", type=int, default=25)
    parser.add_argument("--max-clips", type=int, default=0, help="random subset of clips per dataset")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--no-subtitles", action="store_true")
    own = parse_own_args(parser)

    args, model = build(own.out)
    if dist.get_world_size() != 1:
        raise SystemExit("run e1b with --nproc_per_node 1")
    out = Path(own.out)
    report, lines = {}, ["| dataset | subset | positive accepted | median accepted count | 90% |",
                         "|---|---|---:|---:|---:|"]
    for d_cfg in dataset_configs(args):
        name = d_cfg.dump_name
        if own.only and name not in own.only:
            continue
        start = time.time()
        data = collect(model, build_loader(args, d_cfg, own.max_clips), not own.no_subtitles)
        n = len(data["ids"])
        scores = stage1_scores(model, data)
        k = min(own.pool, n)
        diagonal = scores.diagonal().clone()
        scores.fill_diagonal_(float("-inf"))
        others = scores.topk(k - 1, dim=1).indices
        pool = torch.cat([torch.arange(n)[:, None], others], dim=1)  # column 0 = positive
        scores.diagonal().copy_(diagonal)
        positive_in_stage1_topk = (scores > diagonal[:, None]).sum(1) < k
        subsets = [("v",), ("a",), ("v", "a")] + ([("v", "a", "s")] if "cond_s" in data else [])
        probs = itm_pool(model, data, pool, subsets, own.itm_batch)
        summary, counts = summarize(probs, subsets, own.threshold)
        summary["positive_in_stage1_topk"] = positive_in_stage1_topk.double().mean().item()
        report[name] = summary
        torch.save(dict(ids=data["ids"], pool=pool, probs=probs, subsets=["".join(s) for s in subsets],
                        threshold=own.threshold), out / f"e1b_{name}.pt")
        write_json(out / f"e1b_counts_{name}.json",
                   {clip: {"".join(s): int(counts[i, j]) for j, s in enumerate(subsets)}
                    for i, clip in enumerate(data["ids"])})
        for subset, entry in summary["subsets"].items():
            lines.append(f"| {name} | {subset} | {entry['positive_accept_rate']:.2f} | "
                         f"{entry['accepted_count']['q50']:.1f} | {entry['accepted_count']['q90']:.1f} |")
        print(f"{name}: {n} captions in {time.time() - start:.0f}s; checks {summary['checks']}", flush=True)
    write_json(out / "e1b_summary.json", report)
    (out / "e1b_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
