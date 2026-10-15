"""Euclidean and Lorentz spaces with one interface, so geometry is the only
difference between the two arms of the pilot.

Points are stored as D spatial coordinates; the Lorentz time coordinate is
reconstructed (same convention as utils/lorentz_spectral.py). All Lorentz
arithmetic is FP64.

Both spaces provide: embed (tangent vector at the origin -> point), pairwise and
matched distance, radius (distance to the origin), Gromov product at the origin,
entailment-cone violation (Ganea et al. 2018 for Euclidean, MERU for Lorentz)
and a centroid (mean, or Lorentzian centroid).
"""
import math

import torch
import torch.nn as nn

from utils import lorentz_spectral as lorentz

EPS = 1e-7


class EuclideanSpace(nn.Module):
    name = "euclid"

    def __init__(self, cone_k=0.1):
        super().__init__()
        self.cone_k = cone_k

    def embed(self, tangent):
        return tangent

    def distance(self, x, y):
        return torch.cdist(x, y)

    def matched_distance(self, x, y):
        return (x - y).norm(dim=-1)

    def radius(self, x):
        return x.norm(dim=-1)

    def cone_violation(self, parent, child):
        """Ganea et al. (2018) Euclidean cones, half-aperture asin(K / ||x||)."""
        p_norm = parent.norm(dim=-1).clamp_min(EPS)
        diff = child - parent
        d_norm = diff.norm(dim=-1).clamp_min(EPS)
        aperture = torch.asin((self.cone_k / p_norm).clamp(max=1 - EPS))
        cosine = (child.square().sum(-1) - p_norm.square() - d_norm.square()) / (2 * p_norm * d_norm)
        angle = torch.acos(cosine.clamp(-1 + EPS, 1 - EPS))
        return torch.relu(angle - aperture)

    def centroid(self, points, perron=False):
        return points.mean(dim=-2)


class LorentzSpace(nn.Module):
    name = "lorentz"

    def __init__(self, curvature_init=1.0, learn_curvature=True, cone_k=0.1):
        super().__init__()
        self.log_c = nn.Parameter(torch.tensor(math.log(curvature_init), dtype=torch.float64),
                                  requires_grad=learn_curvature)
        self.cone_k = cone_k

    @property
    def c(self):
        return self.log_c.clamp(math.log(0.1), math.log(10.0)).exp()

    def embed(self, tangent):
        return lorentz.expmap0(tangent, self.c)

    def time(self, x):
        return lorentz.time_component(x.double(), self.c).squeeze(-1)

    def distance(self, x, y):
        return lorentz.pair_distance(x, y, self.c)

    def matched_distance(self, x, y):
        inner = lorentz.matched_inner(x, y, self.c)
        return 2 * torch.asinh(lorentz._sqrt_near_zero((inner - 1) / 2)) / self.c.sqrt()

    def radius(self, x):
        return lorentz.origin_radius(x.double(), self.c)

    def cone_violation(self, parent, child):
        """MERU cone: half-aperture asin(2K / (sqrt(c) ||p||)), exterior angle at p."""
        c = self.c
        parent, child = parent.double(), child.double()
        p_norm = parent.norm(dim=-1).clamp_min(EPS)
        p_time, c_time = self.time(parent), self.time(child)
        aperture = torch.asin((2 * self.cone_k / (c.sqrt() * p_norm)).clamp(max=1 - EPS))
        inner = c * ((parent * child).sum(-1) - p_time * c_time)            # = -cosh(sqrt(c) d) <= -1
        numerator = c_time + inner * p_time
        denominator = p_norm * (inner.square() - 1).clamp_min(EPS).sqrt()
        angle = torch.acos((numerator / denominator).clamp(-1 + EPS, 1 - EPS))
        return torch.relu(angle - aperture)

    def centroid(self, points, perron=False):
        """Lorentzian centroid (uniform weights) or the Perron proxy of the tuple Gram."""
        if perron:
            return lorentz.consensus(points.double(), self.c)
        points = points.double()
        space = points.sum(-2)
        time = self.time(points).sum(-1)
        norm = (self.c * (time.square() - space.square().sum(-1))).clamp_min(EPS).sqrt()
        return space / norm.unsqueeze(-1)


def gromov(space, x, y, matched=False):
    """Gromov product at the origin, (x|y)_O = (r_x + r_y - d(x, y)) / 2."""
    rx, ry = space.radius(x), space.radius(y)
    if matched:
        return (rx + ry - space.matched_distance(x, y)) / 2
    return (rx[:, None] + ry[None, :] - space.distance(x, y)) / 2


def build_space(name, **kwargs):
    if name == "euclid":
        return EuclideanSpace(cone_k=kwargs.get("cone_k", 0.1))
    if name == "lorentz":
        return LorentzSpace(kwargs.get("curvature_init", 1.0), kwargs.get("learn_curvature", True),
                            kwargs.get("cone_k", 0.1))
    raise ValueError(f"unknown geometry {name}")
