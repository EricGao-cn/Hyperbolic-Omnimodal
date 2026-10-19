"""Seven observation subsets: 21 symmetric InfoNCE pairs and nine cone edges.

Lorentz points are stored as spatial coordinates; curvature is fixed at -1.
No backbone imports, so the geometry and distributed objective can be tested on CPU.
"""
import itertools
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

SUBSETS = ("t", "v", "a", "tv", "ta", "va", "tva")
PAIRS = tuple(itertools.combinations(SUBSETS, 2))
EDGES = tuple((s, u) for s in SUBSETS for u in SUBSETS
              if set(s) < set(u) and len(u) == len(s) + 1)
DISJOINT_PAIRS = tuple((s, u) for s, u in PAIRS if not set(s) & set(u))
MINIMAL_PAIRS = DISJOINT_PAIRS + (("tv", "tva"), ("ta", "tva"), ("va", "tva"))
NEW_HOT_HEADS = ("tv", "ta", "tva")
R_MIN, R_MAX, CONE_K = .21, 4., .1


class SubsetHeads(nn.Module):
    def __init__(self, input_dims, dimension=512, geometry="hyperbolic"):
        super().__init__()
        if dimension < 2 or geometry not in ("hyperbolic", "euclidean"):
            raise ValueError("invalid embedding dimension or geometry")
        self.input_dims = dict(input_dims)
        self.dimension, self.geometry = dimension, geometry
        self.direction = nn.ModuleDict({s: nn.Linear(sum(input_dims[m] for m in s), dimension)
                                        for s in SUBSETS})
        self.radius = nn.ModuleDict({s: nn.Linear(dimension, 1) for s in SUBSETS})
        for s in SUBSETS:
            nn.init.xavier_uniform_(self.direction[s].weight)
            nn.init.zeros_(self.direction[s].bias)
            nn.init.zeros_(self.radius[s].weight)
            nn.init.constant_(self.radius[s].bias, math.log((1. - R_MIN) / (R_MAX - 1.)))
        self.compose_initialization(("tv", "ta", "va", "tva"))

    @torch.no_grad()
    def compose_initialization(self, subsets):
        for s in subsets:
            self.direction[s].weight.copy_(torch.cat([self.direction[m].weight for m in s], 1) / len(s))
            self.direction[s].bias.copy_(torch.stack([self.direction[m].bias for m in s]).mean(0))

    @torch.no_grad()
    def initialize_vast(self, state):
        if self.dimension != 512:
            raise ValueError("VAST hot initialization requires dimension=512; use cold init otherwise")
        for s in ("t", "v", "a", "va"):
            stem = "contra_head_" + s + (".linear" if len(s) == 1 else "")
            weight = state[stem + ".weight"]
            if weight.shape != self.direction[s].weight.shape:
                raise ValueError("VAST head shape mismatch: " + stem)
            self.direction[s].weight.copy_(weight)
            bias = state.get(stem + ".bias")
            self.direction[s].bias.copy_(bias if bias is not None else torch.zeros_like(self.direction[s].bias))
        self.compose_initialization(NEW_HOT_HEADS)

    def set_stage(self, stage, initialization="hot"):
        if stage not in ("warmup", "frozen", "joint"):
            raise ValueError(stage)
        self.requires_grad_(True)
        if stage == "warmup":
            self.radius.requires_grad_(False)
            if initialization == "hot":
                for s in ("t", "v", "a", "va"):
                    self.direction[s].requires_grad_(False)

    def forward(self, pooled):
        if any(m not in pooled for m in "tva"):
            raise ValueError("v0 requires complete T/V/A observations")
        result = {}
        # All heads use FP32 even when the backbone is under autocast.
        with torch.autocast(device_type=pooled["t"].device.type, enabled=False):
            for s in SUBSETS:
                h = torch.cat([pooled[m].float() for m in s], -1)
                raw = self.direction[s](h)
                if (raw.norm(dim=-1) <= 1e-12).any():
                    raise FloatingPointError('Zero projection has no valid cone direction: ' + s)
                u = F.normalize(raw, dim=-1, eps=1e-12)
                r = R_MIN + (R_MAX - R_MIN) * self.radius[s](u).double().sigmoid()
                result[s] = u.double() * (r.sinh() if self.geometry == "hyperbolic" else r)
        return result


def time_coordinate(x):
    return (1. + x.square().sum(-1, keepdim=True)).sqrt()


def pair_distance(x, y, geometry="hyperbolic"):
    x, y = x.double(), y.double()
    if geometry == "euclidean":
        delta = (x.square().sum(-1, keepdim=True) + y.square().sum(-1).unsqueeze(0) - 2 * x @ y.T).clamp_min(0)
        return torch.where(delta > 1e-12, delta.clamp_min(1e-12).sqrt(), delta / 1e-6)
    delta = (time_coordinate(x) @ time_coordinate(y).T - x @ y.T - 1.).clamp_min(0)
    # acosh(1+d) = 2 asinh(sqrt(d/2)); finite derivative at coincident points.
    root = torch.where(delta > 1e-12, (delta.clamp_min(1e-12) / 2).sqrt(), delta / math.sqrt(2e-12))
    return 2 * root.asinh()


def cone_violation(parent, child):
    p, q = parent.double(), child.double()
    norm = p.norm(dim=-1)
    tp, tq = time_coordinate(p).squeeze(-1), time_coordinate(q).squeeze(-1)
    inner = (p * q).sum(-1) - tp * tq
    aperture = torch.asin((2 * CONE_K / norm.clamp_min(1e-12)).clamp(max=1. - 1e-12))
    cosine = (tq + tp * inner) / (norm * (inner.square() - 1.).clamp_min(1e-24).sqrt()).clamp_min(1e-12)
    angle = torch.acos(cosine.clamp(-1. + 1e-12, 1. - 1e-12))
    violation = (angle - aperture).relu()
    # Reflexive relation; do not evaluate an undefined exterior angle at the apex.
    return torch.where((p - q).square().sum(-1) <= 1e-24, violation * 0., violation)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.rank = dist.get_rank()
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, 0)

    @staticmethod
    def backward(ctx, grad):
        # Every rank contributes to remote candidate gradients; DDP subsequently
        # averages parameter gradients, matching the global-batch objective.
        grad = grad.contiguous()
        dist.all_reduce(grad)
        return grad.chunk(dist.get_world_size(), 0)[ctx.rank].contiguous()


def gather_candidates(z):
    active = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    packed = torch.stack([z[s] for s in SUBSETS], 1)
    if active:
        size = torch.tensor([len(packed)], device=packed.device)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        if len({int(v.item()) for v in sizes}) != 1:
            raise ValueError("InfoNCE requires equal local batch sizes")
        gathered = _Gather.apply(packed)
        offset = dist.get_rank() * len(packed)
    else:
        gathered, offset = packed, 0
    return {s: gathered[:, j] for j, s in enumerate(SUBSETS)}, offset


def contrast_pairs(mode="all21"):
    if mode == "all21":
        return PAIRS
    if mode == "minimal9":
        return MINIMAL_PAIRS
    raise ValueError("Unknown contrast pair mode: " + str(mode))


def objective(z, temperature=.07, entailment_weight=0., geometry="hyperbolic", distributed=True,
              pair_mode="all21"):
    if temperature <= 0 or entailment_weight < 0:
        raise ValueError("invalid loss weights")
    if geometry == "euclidean" and entailment_weight:
        raise ValueError("Euclidean entailment is out of scope")
    candidates, offset = gather_candidates(z) if distributed else (z, 0)
    target = torch.arange(len(z["t"]), device=z["t"].device) + offset
    by_head = {s: [] for s in SUBSETS}
    pairs = contrast_pairs(pair_mode)
    pair_losses = []
    for s, u in pairs:
        logits = -pair_distance(z[s], candidates[u], geometry) / temperature
        reverse = -pair_distance(z[u], candidates[s], geometry) / temperature
        by_head[s].append(F.cross_entropy(logits, target))
        by_head[u].append(F.cross_entropy(reverse, target))
        pair_losses.append((by_head[s][-1] + by_head[u][-1]) / 2)
    head_losses = {s: torch.stack(values).mean() for s, values in by_head.items()}
    # Preserve the legacy 21-pair reduction exactly. In the irregular 9-pair
    # graph, averaging per-head means would silently reweight individual pairs.
    contrastive = (torch.stack(list(head_losses.values())).mean() if pair_mode == "all21"
                   else torch.stack(pair_losses).mean())
    edges = {s + "->" + u: cone_violation(z[s], z[u]) for s, u in EDGES} if geometry == "hyperbolic" else {}
    entailment = torch.stack(list(edges.values())).mean() if edges else contrastive * 0.
    loss = contrastive + entailment_weight * entailment
    metrics = {"contrastive": contrastive.detach(), "entailment": entailment.detach()}
    metrics.update({"nce_" + s: v.detach() for s, v in head_losses.items()})
    for name, values in edges.items():
        metrics["violation_" + name] = values.detach().mean()
        metrics["satisfied_" + name] = (values.detach() <= 1e-7).double().mean()
    for s in SUBSETS:
        r = z[s].detach().norm(dim=-1)
        metrics["radius_" + s] = (r.asinh() if geometry == "hyperbolic" else r).mean()
    return loss, metrics
