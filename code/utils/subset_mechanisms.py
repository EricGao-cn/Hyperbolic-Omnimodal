"""Controlled head-only mechanism experiments; legacy objectives stay unchanged."""
import itertools
import math
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from utils.subset_hierarchy import (SubsetHeads, SUBSETS, EDGES, R_MIN, contrast_pairs,
                                    pair_distance, time_coordinate, cone_violation)


@dataclass(frozen=True)
class Experiment:
    variant: str = 'B1'
    pair_mode: str = 'minimal9'
    weight: float = .1
    temperature: str = 'fixed'
    detach_radius: bool = False
    compatibility: float = 0.
    eta: float = 1.
    angle_weight: float = 0.
    uncertainty: bool = False
    calibration: float = 0.
    entropy: float = 0.
    radius_max: float = 4.
    training: str = 'standard'

    def __post_init__(self):
        contrast_pairs(self.pair_mode)
        if self.temperature not in ('fixed', 'shared', 'typed'):
            raise ValueError('Unknown temperature mode')
        if self.training not in ('standard', 'layered', 'warmup9', 'no_warmup', 'hierarchy_first', 'ramp', 'radius8', 'warmup21'):
            raise ValueError('Unknown training protocol')
        if not all(math.isfinite(x) and x >= 0 for x in
                   (self.weight, self.compatibility, self.angle_weight, self.calibration, self.entropy)):
            raise ValueError('Invalid loss coefficients')
        if self.eta <= 0 or not math.isfinite(self.eta) or self.radius_max <= 1 or not math.isfinite(self.radius_max):
            raise ValueError('Invalid geometry')


def catalog(pair_mode):
    b = Experiment(pair_mode=pair_mode)
    a1 = replace(b, variant='A1', temperature='shared')
    a2 = replace(a1, variant='A2', temperature='typed')
    a7 = replace(a2, variant='A7', uncertainty=True)
    a8 = replace(a7, variant='A8', calibration=.1)
    return {c.variant: c for c in (
        replace(b, variant='B0', weight=0.), b, replace(b, variant='B2', weight=1.),
        a1, a2, replace(b, variant='A3', detach_radius=True),
        replace(b, variant='A4', compatibility=.1), replace(b, variant='A5', eta=1.2),
        replace(b, variant='A6', angle_weight=.1), a7, a8, replace(a8, variant='A9', entropy=.1))}


def relation(s, t):
    a, b = set(s), set(t)
    return 'disjoint' if not a & b else 'nested' if a < b or b < a else 'overlap'


def cone_parts(parent, child):
    p, q = parent.double(), child.double()
    n = p.norm(dim=-1)
    tp, tq = time_coordinate(p).squeeze(-1), time_coordinate(q).squeeze(-1)
    inner = (p * q).sum(-1) - tp * tq
    aperture = torch.asin((.2 / n.clamp_min(1e-12)).clamp(max=1 - 1e-12))
    cosine = (tq + tp * inner) / (n * (inner.square() - 1).clamp_min(1e-24).sqrt()).clamp_min(1e-12)
    angle = torch.acos(cosine.clamp(-1 + 1e-12, 1 - 1e-12))
    same = (p - q).square().sum(-1) <= 1e-24
    return torch.where(same, angle * 0., angle), aperture


def cap_width(r):
    return torch.asin((.2 / r.tanh() / math.sqrt(1.04)).clamp(-1 + 1e-12, 1 - 1e-12)) - math.atan(.2)


def compatibility_terms(z):
    radii = {s: x.norm(dim=-1).asinh() for s, x in z.items()}
    directions = {s: F.normalize(x, dim=-1) for s, x in z.items()}
    result = {}
    for child in ('tv', 'ta', 'va', 'tva'):
        parents = [p for p, q in EDGES if q == child]
        terms = []
        for s, t in itertools.combinations(parents, 2):
            theta = torch.acos((directions[s] * directions[t]).sum(-1).clamp(-1 + 1e-12, 1 - 1e-12))
            terms.append(theta - cap_width(radii[s]) - cap_width(radii[t]))
        result[child] = torch.stack(terms)
    return result


def uncertainty(x):
    return F.softplus(1 - x.norm(dim=-1).asinh())


class MechanismHeads(SubsetHeads):
    def __init__(self, input_dims, config, dimension=512):
        super().__init__(input_dims, dimension)
        self.config = config

    @torch.no_grad()
    def reset_radii(self, layered=False):
        for s in SUBSETS:
            r = (.5 * len(s)) if layered else 1.
            self.radius[s].weight.zero_()
            self.radius[s].bias.fill_(math.log((r - R_MIN) / (self.config.radius_max - r)))

    def forward(self, pooled):
        result = {}
        for s in SUBSETS:
            raw = self.direction[s](torch.cat([pooled[m].float() for m in s], -1))
            if (raw.norm(dim=-1) <= 1e-12).any():
                raise FloatingPointError('Zero cone direction')
            u = F.normalize(raw, dim=-1, eps=1e-12)
            r = R_MIN + (self.config.radius_max - R_MIN) * self.radius[s](u).double().sigmoid()
            result[s] = u.double() * r.sinh()
        return result


class MechanismLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        names = () if config.temperature == 'fixed' else ('shared',) if config.temperature == 'shared' else ('disjoint', 'nested', 'overlap')
        self.log_tau = nn.ParameterDict({k: nn.Parameter(torch.tensor(math.log(.07), dtype=torch.float64)) for k in names})

    def tau(self, kind):
        if not self.log_tau:
            return .07
        name = 'shared' if self.config.temperature == 'shared' else kind
        return self.log_tau[name].clamp(math.log(.01), math.log(.2)).exp()

    def forward(self, z, hierarchy_scale=1., contrastive=True, edges=None):
        c = self.config
        n = len(z['t'])
        targets = torch.arange(n, device=z['t'].device)
        pairs, types = [], {k: [] for k in ('disjoint', 'nested', 'overlap')}
        zr = {s: F.normalize(x, dim=-1) * x.norm(dim=-1, keepdim=True).detach() for s, x in z.items()} if c.detach_radius else z
        if contrastive:
            for s, t in contrast_pairs(c.pair_mode):
                kind = relation(s, t)
                logits = -pair_distance(zr[s], zr[t]) / self.tau(kind)
                if c.uncertainty and kind == 'nested':
                    parent = s if set(s) < set(t) else t
                    scale = torch.exp(-.5 * uncertainty(z[parent]).detach())
                    logits = logits * (scale[:, None] if parent == s else scale[None, :])
                term = .5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
                pairs.append(term)
                types[kind].append(term)
        zero = z['t'].sum() * 0.
        nce = torch.stack(pairs).mean() if pairs else zero
        residuals, losses, cal = [], [], []
        for s, t in (EDGES if edges is None else edges):
            phi, omega = cone_parts(z[s], z[t])
            residuals.append((phi - omega).relu())
            losses.append((phi - c.eta * omega).relu() + c.angle_weight * phi)
            if c.calibration:
                u = uncertainty(z[s])
                cal.append((phi - omega).relu().detach() * torch.exp(-u) + u)
        entail = torch.stack(losses).mean()
        calibration = torch.stack(cal).mean() if cal else zero
        entropy = zero
        if c.entropy:
            values = []
            for s in SUBSETS[:-1]:
                logp = F.log_softmax(uncertainty(z[s]), dim=0)
                values.append(-(logp.exp() * logp).sum())
            entropy = torch.stack(values).mean()
        caps = compatibility_terms(z) if c.compatibility else None
        compat = torch.cat(list(caps.values())).relu().mean() if caps else zero
        hierarchy = c.weight * entail + c.compatibility * compat + c.calibration * calibration + c.entropy * entropy
        metrics = {'contrastive': nce.detach(), 'entailment': torch.stack(residuals).mean().detach(),
                   'training_entailment': entail.detach(), 'compatibility': compat.detach(),
                   'calibration': calibration.detach(), 'entropy': entropy.detach(), 'hierarchy': hierarchy.detach()}
        metrics.update({'contrastive_' + k: torch.stack(v).mean().detach() for k, v in types.items() if v})
        metrics.update({'tau_' + k: p.clamp(math.log(.01), math.log(.2)).exp().detach() for k, p in self.log_tau.items()})
        return nce + hierarchy_scale * hierarchy, metrics
