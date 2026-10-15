"""Dump per-clip features of one checkpoint for E0, E2 and the E3 pilot.

For every dataset in the config (validation entries; training entries too with
--include-train) this writes <dump-dir>/<dump_name>.rank<r>.pt holding, per
modality t/v/a(/s): pooled backbone features, the raw contrastive projection
before normalization, the model's own retrieval embedding and, for the
direction-radius lift, the log radius multiplier. One caption per clip (the
first), matching the repository's first-caption evaluation.

    torchrun --nproc_per_node 4 experiments/pre_method/extract_features.py \
        --dump-dir OUT/features/k_e0 --config CFG.json --checkpoint CKPT.pt

The config is a normal run.py config; only data_cfg and model_cfg matter.
"""
import argparse
import time

from runtime import build, build_loader, dataset_configs, encode_batch, model_meta, parse_own_args

import torch
import torch.distributed as dist

from common import shard_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--include-train", action="store_true",
                        help="also dump data_cfg.train entries, in evaluation mode")
    parser.add_argument("--only", nargs="*", default=None, help="dump_name values to keep")
    parser.add_argument("--max-items", type=int, default=0,
                        help="random subset of clips per dataset (0 = all)")
    parser.add_argument("--subset-seed", type=int, default=50)
    own = parse_own_args(parser)

    args, model = build(own.dump_dir)
    rank = dist.get_rank()
    meta = model_meta(model, args)
    for d_cfg in dataset_configs(args, own.include_train):
        name = d_cfg.dump_name
        if own.only and name not in own.only:
            continue
        target = shard_path(own.dump_dir, name, rank)
        if target.exists():
            print(f"[rank {rank}] skip {name}: {target} exists", flush=True)
            continue
        start = time.time()
        loader = build_loader(args, d_cfg, own.max_items, own.subset_seed)
        dump = {"ids": [], "captions": [], "pooled": {}, "proj": {}, "emb": {}, "radius_delta": {}}
        for batch in loader:
            batch, out = encode_batch(model, batch)
            dump["ids"] += [str(i) for i in batch.ids]
            dump["captions"] += list(batch.raw_captions)
            for group in ("pooled", "proj", "emb", "radius_delta"):
                for m, value in out[group].items():
                    dump[group].setdefault(m, []).append(value)
        for group in ("pooled", "proj", "emb", "radius_delta"):
            dump[group] = {m: torch.cat(v) for m, v in dump[group].items()}
        dump["meta"] = dict(meta, dataset=name, source_name=d_cfg.name, txt=d_cfg.txt,
                            vision_sample_num=d_cfg.get("vision_sample_num"),
                            audio_sample_num=d_cfg.get("audio_sample_num"),
                            world_size=dist.get_world_size(), rank=rank)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dump, target)
        print(f"[rank {rank}] {name}: {len(dump['ids'])} clips in {time.time() - start:.0f}s -> {target}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
