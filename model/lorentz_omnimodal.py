"""Lorentz spectral alignment with the existing VAST/PMRL backbone and ITM."""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils import lorentz_spectral as geometry
from .pmrl import PMRL
from .hyperbolic_spectral_pmrl import HyperbolicSpectralPMRL


class LorentzOmnimodal(HyperbolicSpectralPMRL):
    """Reuse the paired TV/TA ITM path, replacing only geometry/alignment.

    Deliberately initialize PMRL directly: the scaled-expmap variant has no
    unused direction/radius heads or entailment parameters.
    """

    def __init__(self, config):
        PMRL.__init__(self, config)
        if self.ret_pairwise_stage1_train:
            raise ValueError("Lorentz omnimodal training requires ret%tv%ta")
        if abs(self.lambda_itm - 0.1) > 1e-12:
            raise ValueError("This paired experiment requires lambda_itm=0.1")
        self._init_geometry(config)

    def _init_geometry(self, config):
        self.hyperbolic_mapping = getattr(config, "hyperbolic_mapping", "scaled_expmap")
        self.hyperbolic_score = getattr(config, "hyperbolic_score", "spectral")
        self.hyperbolic_score_transform = getattr(config, "hyperbolic_score_transform", "distance")
        self.hyperbolic_contrast = getattr(config, "hyperbolic_contrast", "text_vs_va")
        self.hyperbolic_spectral_temperature = float(getattr(config, "hyperbolic_temperature", .07))
        self.hyperbolic_consensus_weight = float(getattr(config, "hyperbolic_consensus_weight", 0.))
        self.hyperbolic_entailment_weight = float(getattr(config, "hyperbolic_entailment_weight", 0.))
        self.hyperbolic_spectral_anchor = "t"
        if self.hyperbolic_mapping not in ("scaled_expmap", "direction_radius"):
            raise ValueError("Unknown hyperbolic_mapping")
        if self.hyperbolic_score not in ("spectral", "uniform", "allpair"):
            raise ValueError("Unknown hyperbolic_score")
        if self.hyperbolic_score_transform not in geometry.SCORE_TRANSFORMS:
            raise ValueError("Unknown hyperbolic_score_transform")
        if self.hyperbolic_contrast not in ("text_vs_va", "symmetric"):
            raise ValueError("Unknown hyperbolic_contrast")
        if self.hyperbolic_score == "allpair" and (self.hyperbolic_score_transform != "distance"
                                                   or self.hyperbolic_contrast != "text_vs_va"):
            raise ValueError("allpair is the geodesic control; it takes no transform or contrast mode")
        if not math.isfinite(self.hyperbolic_spectral_temperature) or self.hyperbolic_spectral_temperature <= 0:
            raise ValueError("hyperbolic_temperature must be finite and positive")
        if not math.isfinite(self.hyperbolic_consensus_weight) or self.hyperbolic_consensus_weight < 0:
            raise ValueError("hyperbolic_consensus_weight must be finite and nonnegative")
        dimension = config.contra_dim
        self._init_entailment(config, dimension)
        initial_c = float(getattr(config, "hyperbolic_curvature_init", 1.))
        if not .1 <= initial_c <= 10.:
            raise ValueError("Initial curvature must lie in [0.1,10]")
        learn_c = bool(getattr(config, "hyperbolic_learn_curvature", True))
        self.hyperbolic_log_curvature = nn.Parameter(torch.tensor(math.log(initial_c), dtype=torch.float64), requires_grad=learn_c)
        self.register_buffer("_hyperbolic_log_curvature_min", torch.tensor(math.log(.1), dtype=torch.float64))
        self.register_buffer("_hyperbolic_log_curvature_max", torch.tensor(math.log(10.), dtype=torch.float64))
        multiplier = float(getattr(config, "hyperbolic_scale_multiplier", 1.))
        initial_scale = multiplier / math.sqrt(dimension) if self.hyperbolic_mapping == "scaled_expmap" else 1.
        if not 0 < initial_scale <= 1:
            raise ValueError("Initial scale must lie in (0,1]")
        self.hyperbolic_log_modality_scales = nn.ParameterDict({
            m: nn.Parameter(torch.tensor(math.log(initial_scale), dtype=torch.float64)) for m in self.modalities
        })
        if self.hyperbolic_mapping == "direction_radius":
            self.hyperbolic_direction_heads = nn.ModuleDict({m: nn.Linear(dimension, dimension, bias=False) for m in self.modalities})
            self.hyperbolic_radius_heads = nn.ModuleDict({m: nn.Linear(dimension, 1) for m in self.modalities})
            for head in self.hyperbolic_direction_heads.values():
                nn.init.eye_(head.weight)
            for head in self.hyperbolic_radius_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        subsets = [s for k in (1, 2, 3) for s in itertools.combinations(range(3), k)]
        self._subset_pairs = [(a, b) for a, b in itertools.combinations(subsets, 2) if set(a) | set(b) == {0, 1, 2}]
        self._subset_rng = torch.Generator().manual_seed(int(getattr(config, "hyperbolic_subset_seed", 50)) + 7419)
        self._geometry_diagnostics = {}

    def _init_entailment(self, config, dimension):
        """Optional cone entailment; the 0929 suite (weight 0) is unchanged.

        perron: the parent is the Lorentz Perron proxy of the same Gram matrix
        that defines the score. Other parents reuse the kernel-model definitions.
        """
        if not math.isfinite(self.hyperbolic_entailment_weight) or self.hyperbolic_entailment_weight < 0:
            raise ValueError("hyperbolic_entailment_weight must be finite and nonnegative")
        self.hyperbolic_entailment_parent = getattr(config, "hyperbolic_entailment_parent", "modality_edges")
        if self.hyperbolic_entailment_parent not in ("modality_edges", "perron", "tangent_mean", "svd_learned_radius"):
            raise ValueError("unknown hyperbolic_entailment_parent")
        self.hyperbolic_entailment_min_radius = float(getattr(config, "hyperbolic_entailment_min_radius", .1))
        self.hyperbolic_parent_contraction = float(getattr(config, "hyperbolic_parent_contraction", 1.))
        self.hyperbolic_parent_floor_margin = float(getattr(config, "hyperbolic_parent_floor_margin", 1e-3))
        self.hyperbolic_entailment_edges = self._normalize_entailment_edges(
            getattr(config, "hyperbolic_entailment_edges", self.default_entailment_edges))
        if self.hyperbolic_entailment_min_radius <= 0 or self.hyperbolic_parent_floor_margin <= 0:
            raise ValueError("cone radius and parent floor margin must be positive")
        if not 0 < self.hyperbolic_parent_contraction <= 1:
            raise ValueError("parent contraction must be in (0, 1]")
        if self.hyperbolic_entailment_weight and self.hyperbolic_entailment_parent == "svd_learned_radius":
            with torch.random.fork_rng(devices=[]):
                self.hyperbolic_parent_radius_head = nn.Linear(dimension, 1)
            nn.init.zeros_(self.hyperbolic_parent_radius_head.weight)
            nn.init.zeros_(self.hyperbolic_parent_radius_head.bias)

    def _lift(self, features, modality):
        with autocast(enabled=False):
            curvature = self.hyperbolic_curvature
            if self.hyperbolic_mapping == "direction_radius":
                unit = F.normalize(features.float(), dim=-1)
                direction = F.normalize(self.hyperbolic_direction_heads[modality](unit), dim=-1)
                delta = math.log(3.) * torch.tanh(self.hyperbolic_radius_heads[modality](unit))
                tangent = direction.double() * (self.hyperbolic_log_modality_scales[modality] + delta.double()).exp()
            else:
                tangent = features.double() * self.hyperbolic_log_modality_scales[modality].exp()
            points = geometry.expmap0(tangent, curvature)
            if self.training:
                with torch.no_grad():
                    self._geometry_diagnostics[f"radius_{modality}"] = geometry.origin_radius(points, curvature).mean().item()
                    self._geometry_diagnostics[f"scale_{modality}"] = self.hyperbolic_log_modality_scales[modality].exp().item()
                    clipped = curvature.sqrt() * tangent.norm(dim=-1) >= geometry.MAX_SCALED_RADIUS
                    self._geometry_diagnostics[f"radius_clamp_fraction_{modality}"] = clipped.double().mean().item()
            return points

    def _pairwise_distance(self, left, right):
        with autocast(enabled=False):
            return geometry.pair_distance(left, right, self.hyperbolic_curvature)

    def retrieval_similarity(self, text, candidate):
        return -self._pairwise_distance(text, candidate)

    def retrieval_tuple_similarity(self, text, candidates, chunk_size=64):
        """Native T-to-VA scores; subtitles remain an ITM-only condition."""
        if set(candidates) != {"v", "a"}:
            raise ValueError("Native tuple retrieval requires paired video and audio")
        with autocast(enabled=False):
            c = self.hyperbolic_curvature
            score = self.hyperbolic_score
            if score == "allpair":
                # A descriptive pairwise fusion only; not used in the main panel.
                return (self.retrieval_similarity(text, candidates["v"])
                        + self.retrieval_similarity(text, candidates["a"])) / 2
            # Bound both axes: 64 queries against a full corpus can exceed the
            # CUDA batched-eigensolver limit even when GPU memory is available.
            rows = []
            for q in text.split(chunk_size):
                columns = []
                candidate_chunk = max(1, 1024 // q.shape[0])
                for v, a in zip(candidates["v"].split(candidate_chunk), candidates["a"].split(candidate_chunk)):
                    columns.append(geometry.gram_similarity(
                        geometry.candidate_gram(q, [v, a], c), c, score,
                        self.hyperbolic_score_transform))
                rows.append(torch.cat(columns, dim=1))
            return torch.cat(rows, dim=0)

    def _geodesic_contrastive(self, left, right):
        all_left, targets = self._distributed_batch(left)
        all_right, _ = self._distributed_batch(right)
        temperature = self.hyperbolic_spectral_temperature
        return .5 * (F.cross_entropy(-self._pairwise_distance(left, all_right) / temperature, targets)
                     + F.cross_entropy(-self._pairwise_distance(right, all_left) / temperature, targets))

    def spectral_loss(self, text, vision, audio):
        with autocast(enabled=False):
            if self.hyperbolic_score == "allpair":
                return (self._geodesic_contrastive(text, vision)
                        + self._geodesic_contrastive(text, audio)
                        + self._geodesic_contrastive(vision, audio)) / 3
            local = [text, vision, audio]
            gathered = [self._distributed_batch(x)[0] for x in local]
            targets = self._distributed_batch(text)[1]
            c = self.hyperbolic_curvature
            # lambda_max is permutation invariant, so any modality can be the query row.
            anchors = (0,) if self.hyperbolic_contrast == "text_vs_va" else (0, 1, 2)
            losses = []
            for anchor in anchors:
                rest = [m for m in range(3) if m != anchor]
                forward = self._tuple_similarity(local[anchor], [gathered[m] for m in rest], c)
                backward = self._tuple_similarity(gathered[anchor], [local[m] for m in rest], c).T
                if anchor == 0:
                    with torch.no_grad():
                        self._geometry_diagnostics["positive_tuple_similarity"] = forward.gather(1, targets[:, None]).mean().item()
                        self._geometry_diagnostics["candidate_tuple_similarity"] = forward.mean().item()
                        if self.hyperbolic_score_transform == "distance":  # 0929 key names
                            self._geometry_diagnostics["positive_tuple_distance"] = -self._geometry_diagnostics["positive_tuple_similarity"]
                            self._geometry_diagnostics["candidate_tuple_distance"] = -self._geometry_diagnostics["candidate_tuple_similarity"]
                losses.append(.5 * (F.cross_entropy(forward / self.hyperbolic_spectral_temperature, targets)
                                    + F.cross_entropy(backward / self.hyperbolic_spectral_temperature, targets)))
            if self.training:
                self._geometry_diagnostics.update(geometry.perron_diagnostics(torch.stack(local, dim=-2), c))
            return torch.stack(losses).mean()

    def _tuple_similarity(self, query, candidates, curvature):
        return geometry.gram_similarity(geometry.candidate_gram(query, candidates, curvature), curvature,
                                        self.hyperbolic_score, self.hyperbolic_score_transform)

    def perron_parent(self, text, vision, audio):
        """On-manifold Perron proxy, pushed out to the cone's domain floor when inside it."""
        with autocast(enabled=False):
            c = self.hyperbolic_curvature.double()
            parent = geometry.consensus(torch.stack([text, vision, audio], dim=-2).double(), c)
            floor = math.asinh(2 * self.hyperbolic_entailment_min_radius * (1 + self.hyperbolic_parent_floor_margin))
            floor = torch.as_tensor(floor, dtype=parent.dtype, device=parent.device) / c.sqrt()
            norm = parent.norm(dim=-1, keepdim=True)
            radius = torch.asinh(c.sqrt() * norm) / c.sqrt()
            active = radius < floor
            floored = parent / norm.clamp_min(1e-12) * torch.sinh(c.sqrt() * floor) / c.sqrt()
            if self.training:
                self._geometry_diagnostics["parent_radius_mean"] = radius.mean().item()
                self._geometry_diagnostics["parent_floor_active_fraction"] = active.double().mean().item()
            return torch.where(active, floored, parent)

    def entailment_loss(self, text, vision, audio):
        with autocast(enabled=False):
            points = {"t": text.float(), "v": vision.float(), "a": audio.float()}
            if self.hyperbolic_entailment_parent == "modality_edges":
                if not self.hyperbolic_entailment_edges:
                    raise ValueError("modality_edges entailment needs hyperbolic_entailment_edges")
                edges = [(points[p], points[ch], p + ch) for p, ch in self.hyperbolic_entailment_edges]
            else:
                if self.hyperbolic_entailment_parent == "perron":
                    parent = self.perron_parent(text, vision, audio).float()
                else:
                    parent = self.common_parent(points["t"], points["v"], points["a"])
                edges = [(parent, points[m], "p" + m) for m in self.modalities]
            violations = [(name, self._entailment_violation(p, ch)) for p, ch, name in edges]
            if self.training:
                with torch.no_grad():
                    for name, value in violations:
                        self._geometry_diagnostics[f"cone_satisfied_{name}"] = (value <= 0).double().mean().item()
            return torch.stack([value.mean() for _, value in violations]).mean()

    def consensus_loss(self, text, vision, audio):
        with autocast(enabled=False):
            pair_index = int(torch.randint(len(self._subset_pairs), (), generator=self._subset_rng))
            subset_a, subset_b = self._subset_pairs[pair_index]
            points = torch.stack([text, vision, audio], dim=-2)
            c = self.hyperbolic_curvature
            first = geometry.consensus(points[:, subset_a, :], c)
            second = geometry.consensus(points[:, subset_b, :], c)
            self._geometry_diagnostics["consensus_subset_pair"] = pair_index
            return self._geodesic_contrastive(first, second)

    def forward_ret(self, batch, task, compute_loss=True):
        if compute_loss and task != "ret%tv%ta":
            raise ValueError("Paired training requires exactly ret%tv%ta")
        result = super().forward_ret(batch, task, compute_loss)
        if compute_loss:
            result["loss_lorentz_alignment"] = result.pop("loss_hyperbolic_spectral")
            if self.hyperbolic_consensus_weight:
                cons = self.consensus_loss(batch["feat_t"], batch["feat_v"], batch["feat_a"])
                result["loss_lorentz_consensus"] = self.hyperbolic_consensus_weight * cons
                self._geometry_diagnostics["consensus_unweighted"] = cons.detach().item()
        return result

    def training_diagnostics(self):
        result = dict(self._geometry_diagnostics)
        result["curvature"] = self.hyperbolic_curvature.detach().item()
        return result
