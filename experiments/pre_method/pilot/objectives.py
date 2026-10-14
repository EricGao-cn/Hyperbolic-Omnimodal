"""Pilot objectives O1-O4 (E3) and the order/text-placement variants (E4).

O1 allpair   symmetric InfoNCE between every pair of single modalities.
O2 collapse  the collapse family: PMRL (unit vectors, singular-value softmax intra
             plus leading-singular-vector inter) in Euclidean space; the M-matrix
             tuple InfoNCE with score -D_M (text vs paired v,a) in Lorentz space.
O3 subsets   symmetric InfoNCE between h_i(S) and h_j(S') for pairs of distinct
             subsets, instance i positive for j = i.
O4 order     O3 plus entailment cones over covering pairs S c S' (|S'| = |S| + 1):
             join (default): h(S) => h(S')  (fewer modalities = more generic parent);
             meet:           h(S') => h(S)  (the earlier proxy -> modality direction).

text_mode=query keeps text out of the subset lattice: subsets range over the
clip modalities, text is contrasted with each clip subset, and the order term
adds text => h(full clip) (MERU direction).
"""
import random

import torch
import torch.nn.functional as F

from utils import lorentz_spectral as lorentz

from pilot.geometry import gromov
from pilot.model import all_subsets

OBJECTIVES = ("allpair", "collapse", "subsets", "order")


def similarity(space, x, y, score):
    if score == "dist":
        return -space.distance(x, y)
    if score == "gromov":
        return gromov(space, x, y)
    if score == "cos":
        return F.normalize(x.float(), dim=-1) @ F.normalize(y.float(), dim=-1).T
    raise ValueError(score)


def info_nce(logits):
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


class Objective:
    def __init__(self, name, modalities, score="dist", order="join", text_mode="lattice",
                 order_weight=0.1, pairs_per_step=0, tau1=0.5, tau2=0.1, seed=50):
        if name not in OBJECTIVES:
            raise ValueError(name)
        self.name, self.score, self.order, self.text_mode = name, score, order, text_mode
        self.modalities = tuple(modalities)
        self.order_weight = order_weight
        self.pairs_per_step = pairs_per_step
        self.tau1, self.tau2 = tau1, tau2
        self.rng = random.Random(seed)
        clip = tuple(m for m in self.modalities if m != "t")
        if text_mode == "query":
            self.subsets = all_subsets(clip)
            self.pairs = [(("t",), s) for s in self.subsets] + [
                (a, b) for i, a in enumerate(self.subsets) for b in self.subsets[i + 1:]]
        else:
            self.subsets = all_subsets(self.modalities)
            self.pairs = [(a, b) for i, a in enumerate(self.subsets) for b in self.subsets[i + 1:]]
        self.covering = [(a, b) for a in self.subsets for b in self.subsets
                         if len(b) == len(a) + 1 and set(a) < set(b)]
        self.trains_subsets = name in ("subsets", "order")

    # ------------------------------------------------------------------ losses
    def __call__(self, model, features, logit_scale):
        if self.name == "allpair":
            return self.allpair(model, features, logit_scale)
        if self.name == "collapse":
            return self.collapse(model, features, logit_scale)
        return self.subset_loss(model, features, logit_scale)

    def allpair(self, model, features, logit_scale):
        points = {m: model(features, (m,)) for m in self.modalities}
        losses = [info_nce(logit_scale * similarity(model.space, points[a], points[b], self.score))
                  for i, a in enumerate(self.modalities) for b in self.modalities[i + 1:]]
        loss = torch.stack(losses).mean()
        return loss, {"loss_contrast": loss.item()}

    def collapse(self, model, features, logit_scale):
        points = {m: model(features, (m,)) for m in self.modalities}
        if model.space.name == "lorentz":
            c = model.space.c
            rest = [points[m] for m in self.modalities if m != "t"]
            forward = lorentz.gram_similarity(lorentz.candidate_gram(points["t"], rest, c), c, "spectral", "distance")
            loss = info_nce(logit_scale * forward)
            return loss, {"loss_contrast": loss.item()}
        # PMRL on unit vectors: rows of Z are the modalities of one instance
        z = torch.stack([F.normalize(points[m].float(), dim=-1) for m in self.modalities], dim=1)   # [B, k, D]
        _, sigma, vh = torch.linalg.svd(z, full_matrices=False)                                     # vh: [B, k, D]
        lead = vh[:, 0]
        lead = torch.where((lead * z.mean(1)).sum(-1, keepdim=True) < 0, -lead, lead)   # PMRL sign convention
        intra = F.cross_entropy(sigma / self.tau1, torch.zeros(len(sigma), dtype=torch.long, device=sigma.device))
        inter = F.cross_entropy(lead @ lead.T / self.tau2, torch.arange(len(lead), device=lead.device))
        loss = intra + inter
        return loss, {"loss_intra": intra.item(), "loss_inter": inter.item()}

    def subset_loss(self, model, features, logit_scale):
        pairs = self.pairs
        if self.pairs_per_step and self.pairs_per_step < len(pairs):
            pairs = self.rng.sample(pairs, self.pairs_per_step)
        needed = {s for pair in pairs for s in pair}
        if self.name == "order":
            needed |= {s for pair in self.covering for s in pair}
            if self.text_mode == "query":
                needed |= {("t",), self.subsets[-1]}
        points = {s: model(features, s) for s in needed}
        contrast = torch.stack([info_nce(logit_scale * similarity(model.space, points[a], points[b], self.score))
                                for a, b in pairs]).mean()
        logs = {"loss_contrast": contrast.item()}
        loss = contrast
        if self.name == "order":
            edges = self.covering if self.order == "join" else [(b, a) for a, b in self.covering]
            if self.text_mode == "query":
                edges = edges + [(("t",), self.subsets[-1])]
            violations = [model.space.cone_violation(points[p], points[ch]) for p, ch in edges]
            order = torch.stack([v.mean() for v in violations]).mean()
            loss = loss + self.order_weight * order
            logs["loss_order"] = order.item()
            logs["cone_satisfied"] = torch.stack([(v <= 0).double().mean() for v in violations]).mean().item()
        return loss, logs

    # --------------------------------------------------------------- scoring
    def retrieval_scores(self, model, points, query, candidate):
        """Score matrix for evaluation. Objectives that never train subset
        embeddings score a multi-modality candidate by the mean single scores."""
        if self.trains_subsets or len(candidate) == 1 and len(query) == 1:
            return similarity(model.space, points[query], points[candidate], self.score)
        scores = [similarity(model.space, points[(q,)], points[(m,)], self.score) for q in query for m in candidate]
        return torch.stack(scores).mean(0)
