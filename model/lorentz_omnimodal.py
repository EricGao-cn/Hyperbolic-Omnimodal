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
        self.hyperbolic_spectral_temperature = float(getattr(config, "hyperbolic_temperature", .07))
        self.hyperbolic_consensus_weight = float(getattr(config, "hyperbolic_consensus_weight", 0.))
        self.hyperbolic_entailment_weight = float(getattr(config, "hyperbolic_entailment_weight", 0.))
        self.hyperbolic_spectral_anchor = "t"
        if self.hyperbolic_mapping not in ("scaled_expmap", "direction_radius"):
            raise ValueError("Unknown hyperbolic_mapping")
        if self.hyperbolic_score not in ("spectral", "uniform", "allpair"):
            raise ValueError("Unknown hyperbolic_score")
        if not math.isfinite(self.hyperbolic_spectral_temperature) or self.hyperbolic_spectral_temperature <= 0:
            raise ValueError("hyperbolic_temperature must be finite and positive")
        if not math.isfinite(self.hyperbolic_consensus_weight) or self.hyperbolic_consensus_weight < 0:
            raise ValueError("hyperbolic_consensus_weight must be finite and nonnegative")
        if self.hyperbolic_entailment_weight != 0:
            raise ValueError("This suite excludes entailment")
        dimension = config.contra_dim
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
            return torch.cat([
                -geometry.gram_distance(geometry.candidate_gram(q, [candidates["v"], candidates["a"]], c), c, score)
                for q in text.split(chunk_size)
            ], dim=0)

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
            all_text, targets = self._distributed_batch(text)
            all_vision, _ = self._distributed_batch(vision)
            all_audio, _ = self._distributed_batch(audio)
            c = self.hyperbolic_curvature
            forward = -geometry.gram_distance(geometry.candidate_gram(text, [all_vision, all_audio], c), c, self.hyperbolic_score)
            backward = -geometry.gram_distance(geometry.candidate_gram(all_text, [vision, audio], c), c, self.hyperbolic_score).T
            with torch.no_grad():
                self._geometry_diagnostics["positive_tuple_distance"] = -forward.gather(1, targets[:, None]).mean().item()
                self._geometry_diagnostics["candidate_tuple_distance"] = -forward.mean().item()
            return .5 * (F.cross_entropy(forward / self.hyperbolic_spectral_temperature, targets)
                         + F.cross_entropy(backward / self.hyperbolic_spectral_temperature, targets))

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
