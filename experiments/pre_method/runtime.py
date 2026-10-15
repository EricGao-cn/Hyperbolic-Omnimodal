"""Build the repo's model and evaluation loaders outside run.py.

Scripts accept the normal run.py arguments (--config, --checkpoint,
--pretrain_dir, ...) plus their own flags, which are parsed first and removed
from sys.argv so utils.args.get_args() sees only what it knows.

Launch with torchrun from the repository root, e.g.
    torchrun --nproc_per_node 4 experiments/pre_method/extract_features.py --config CFG ...
"""
import copy
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402

LIFTED_MODEL_TYPES = ("hyperbolic_spectral_pmrl", "lorentz_omnimodal")


def parse_own_args(parser):
    """Parse this script's flags, leave the rest of sys.argv for get_args()."""
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return own


def build(own_output_dir):
    from utils.args import get_args
    from utils.build_model import build_model
    from utils.initialize import initialize

    args = get_args()
    args.run_cfg.output_dir = str(Path(own_output_dir) / "_run")
    args.run_cfg.mode = "testing"
    initialize(args)
    model, _, _ = build_model(args)
    model = getattr(model, "module", model)
    model.eval()
    return args, model


def dataset_configs(args, include_train=False):
    """Validation entries plus, optionally, training entries forced to eval mode.

    An entry may carry `dump_name` to distinguish two entries that share the
    dataset `name` (for example the VAST train and holdout splits, which must
    both be named finetune_area for the media filter in AnnoIndexedDataset).
    """
    entries = [copy.deepcopy(d) for d in args.data_cfg.val]
    if include_train:
        for d in args.data_cfg.train:
            d = copy.deepcopy(d)
            d.training = False
            if "vision_transforms" in d:
                d.vision_transforms = "none"
            if "dump_name" not in d:
                d.dump_name = f"{d.name}_train"
            entries.append(d)
    for d in entries:
        if "dump_name" not in d:  # EasyDict: set attributes, not dict.setdefault
            d.dump_name = d.name
    return entries


def build_loader(args, d_cfg, max_items=0, seed=50):
    from data import data_registry
    from data.loader import PrefetchLoader
    from utils.build_dataloader import build_dataloader

    dataset = data_registry[d_cfg.type](d_cfg, args)
    dataset.name = d_cfg.name
    if max_items and len(dataset.annos_new) > max_items:
        keep = sorted(random.Random(seed).sample(range(len(dataset.annos_new)), max_items))
        dataset.annos_new = [dataset.annos_new[i] for i in keep]
        dataset.annos = dataset.annos_new
        dataset.idx = list(range(len(dataset.annos_new)))
    loader = build_dataloader(dataset, dataset.collate_fn, False, d_cfg.batch_size,
                              d_cfg.n_workers, dataset.worker_init_fn, dataset.use_sampler)
    return PrefetchLoader(loader)


def first_captions(raw_captions):
    return [c[0] if isinstance(c, list) else c for c in raw_captions]


@torch.no_grad()
def encode_batch(model, batch):
    """Pooled, raw-projection and final retrieval features for one batch."""
    batch = edict(batch)
    batch.raw_captions = first_captions(batch.raw_captions)
    out = {"pooled": {}, "proj": {}, "emb": {}, "radius_delta": {}}
    sources = {
        "t": ("caption_output", model.pool_text_for_contra, model.contra_head_t),
        "v": ("vision_output", model.pool_vision_for_contra, model.contra_head_v),
        "a": ("audio_output", model.pool_audio_for_contra, model.contra_head_a),
    }
    if getattr(batch, "raw_subtitles", None) is not None:
        sources["s"] = ("subtitle_output", model.pool_text_for_contra, model.contra_head_s)
    for m, (key, pool, head) in sources.items():
        pooled = pool(model.batch_get(batch, key))
        proj = head(pooled)
        out["pooled"][m] = pooled.float().cpu()
        out["proj"][m] = proj.float().cpu()
        out["emb"][m] = model.batch_get(batch, f"feat_{m}").double().cpu()
        heads = getattr(model, "hyperbolic_radius_heads", None)
        if heads is not None and m in heads:
            unit = F.normalize(proj.float(), dim=-1)
            out["radius_delta"][m] = (math.log(3.0) * torch.tanh(heads[m](unit).squeeze(-1))).float().cpu()
    return batch, out


def model_meta(model, args):
    model_type = args.model_cfg.model_type
    meta = dict(model_type=model_type, contra_dim=args.model_cfg.contra_dim,
                checkpoint=args.run_cfg.checkpoint or "", pretrain_dir=args.run_cfg.pretrain_dir or "",
                curvature=None, lifted={}, log_modality_scales={},
                mapping=getattr(model, "hyperbolic_mapping", "direction_radius"
                                if model_type == "hyperbolic_spectral_pmrl" else None))
    if model_type in LIFTED_MODEL_TYPES:
        meta["curvature"] = float(model.hyperbolic_curvature.detach().double().item())
        meta["lifted"] = {m: True for m in ("t", "v", "a")}
        meta["log_modality_scales"] = {m: float(p.detach().item())
                                       for m, p in model.hyperbolic_log_modality_scales.items()}
    return meta
