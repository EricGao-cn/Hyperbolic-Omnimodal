"""Head-only subset encoder shared by every pilot cell.

Each modality's cached pooled feature becomes one token (linear projection plus
a modality-type embedding). A two-layer transformer runs over the tokens of a
subset S with absent modalities masked, the present tokens are mean-pooled, and
a linear head gives a tangent vector that the space maps to a point h(S).

Encoder variants (E4):
- fusion:   h(S) from the transformer over S's tokens (default; can represent a join);
- centroid: h({m}) from the transformer, h(S) = centroid of the singletons;
- perron:   as centroid, with the Lorentz Perron proxy (mean in Euclidean space).
Centroids sit nearer the origin than their members, i.e. at the meet; the
pilot tests whether that blocks the subset -> superset order.
"""
import itertools
import math

import torch
import torch.nn as nn


def all_subsets(modalities):
    return [tuple(s) for k in range(1, len(modalities) + 1) for s in itertools.combinations(modalities, k)]


class SubsetEncoder(nn.Module):
    def __init__(self, input_dims, modalities, space, width, encoder="fusion", d_model=512,
                 layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.modalities = tuple(modalities)
        self.space = space
        self.encoder = encoder
        self.inputs = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(input_dims[m]), nn.Linear(input_dims[m], d_model))
                                     for m in self.modalities})
        self.type_embedding = nn.ParameterDict({m: nn.Parameter(0.02 * torch.randn(d_model)) for m in self.modalities})
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, width)
        nn.init.normal_(self.head.weight, std=1 / math.sqrt(d_model))
        nn.init.zeros_(self.head.bias)
        # tangent norm starts near 1 in both geometries; no upper clamp at 1
        self.log_scale = nn.Parameter(torch.tensor(-0.5 * math.log(width)))

    def tangent(self, features, subset):
        tokens = torch.stack([self.inputs[m](features[m].float()) + self.type_embedding[m] for m in subset], dim=1)
        hidden = self.norm(self.transformer(tokens)).mean(dim=1)
        return self.head(hidden) * self.log_scale.clamp(max=math.log(10.0)).exp()

    def forward(self, features, subset):
        """Point h(S) for a batch; `features` maps modality -> [B, d_m]."""
        subset = tuple(m for m in self.modalities if m in subset)
        if self.encoder == "fusion" or len(subset) == 1:
            return self.space.embed(self.tangent(features, subset))
        singles = torch.stack([self.space.embed(self.tangent(features, (m,))) for m in subset], dim=-2)
        return self.space.centroid(singles, perron=self.encoder == "perron")

    def encode_all(self, features, subsets):
        return {s: self(features, s) for s in subsets}
