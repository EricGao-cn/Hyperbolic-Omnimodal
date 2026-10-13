import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.distributed import all_gather_with_grad, concat_all_gather

from .pmrl import PMRL
from .svd_proxy import svd_proxy


class HyperbolicSpectralPMRL(PMRL):
    """PMRL with only the direction/radius Lorentz module and spectral loss."""

    modalities = ("t", "v", "a")
    modality_aliases = {
        "t": "t",
        "text": "t",
        "v": "v",
        "video": "v",
        "vision": "v",
        "a": "a",
        "audio": "a",
    }
    default_entailment_edges = (("t", "v"), ("t", "a"))

    def __init__(self, config):
        super().__init__(config)
        dimension = config.contra_dim
        self.hyperbolic_spectral_temperature = float(
            getattr(config, "hyperbolic_spectral_temperature", 0.02)
        )
        self.hyperbolic_entailment_weight = float(
            getattr(config, "hyperbolic_entailment_weight", 0.0)
        )
        self.hyperbolic_entailment_min_radius = float(
            getattr(config, "hyperbolic_entailment_min_radius", 0.1)
        )
        self.hyperbolic_entailment_parent = getattr(config, "hyperbolic_entailment_parent", "modality_edges")
        self.hyperbolic_parent_contraction = float(getattr(config, "hyperbolic_parent_contraction", 0.5))
        self.hyperbolic_parent_floor_margin = float(getattr(config, "hyperbolic_parent_floor_margin", 1e-3))
        if self.hyperbolic_entailment_parent not in ("modality_edges", "tangent_mean", "svd_mean_radius", "svd_learned_radius"):
            raise ValueError("unknown hyperbolic_entailment_parent")
        if not 0 < self.hyperbolic_parent_contraction <= 1:
            raise ValueError("parent contraction must be in (0, 1]")
        if self.hyperbolic_parent_floor_margin <= 0:
            raise ValueError("parent cone-domain margin must be positive")
        self.hyperbolic_spectral_anchor = self._normalize_modality(
            getattr(config, "hyperbolic_spectral_anchor", "text"),
            field="hyperbolic_spectral_anchor",
        )
        self.hyperbolic_entailment_edges = self._normalize_entailment_edges(
            getattr(
                config,
                "hyperbolic_entailment_edges",
                self.default_entailment_edges,
            )
        )
        if self.hyperbolic_spectral_temperature <= 0:
            raise ValueError("hyperbolic_spectral_temperature must be positive")
        if self.hyperbolic_entailment_weight < 0:
            raise ValueError("hyperbolic_entailment_weight must be non-negative")
        if self.hyperbolic_entailment_min_radius <= 0:
            raise ValueError("hyperbolic_entailment_min_radius must be positive")
        self.hyperbolic_direction_heads = nn.ModuleDict(
            {
                modality: nn.Linear(dimension, dimension, bias=False)
                for modality in self.modalities
            }
        )
        self.hyperbolic_radius_heads = nn.ModuleDict(
            {
                modality: nn.Linear(dimension, 1)
                for modality in self.modalities
            }
        )
        for head in self.hyperbolic_direction_heads.values():
            nn.init.eye_(head.weight)
        for head in self.hyperbolic_radius_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.hyperbolic_log_modality_scales = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.zeros(()))
                for modality in self.modalities
            }
        )
        if self.hyperbolic_entailment_parent == "svd_learned_radius":
            # Preserve the global RNG sequence and zero-init to the fixed-radius arm.
            with torch.random.fork_rng(devices=[]):
                self.hyperbolic_parent_radius_head = nn.Linear(dimension, 1)
            nn.init.zeros_(self.hyperbolic_parent_radius_head.weight)
            nn.init.zeros_(self.hyperbolic_parent_radius_head.bias)
        self.hyperbolic_log_curvature = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "_hyperbolic_log_curvature_min", torch.tensor(math.log(0.1))
        )
        self.register_buffer(
            "_hyperbolic_log_curvature_max", torch.tensor(math.log(10.0))
        )

    @classmethod
    def _normalize_modality(cls, value, field="modality"):
        if not isinstance(value, str):
            raise ValueError("{} must be a string".format(field))
        normalized = cls.modality_aliases.get(value.strip().lower())
        if normalized is None:
            raise ValueError(
                "{} must be one of text, video/vision, or audio; got {!r}".format(
                    field, value
                )
            )
        return normalized

    @classmethod
    def _normalize_entailment_edges(cls, edges):
        if edges is None:
            return cls.default_entailment_edges
        if isinstance(edges, (str, bytes)):
            raise ValueError(
                "hyperbolic_entailment_edges must be a list of [parent, child] pairs"
            )

        normalized = []
        seen = set()
        try:
            raw_edges = list(edges)
        except TypeError as error:
            raise ValueError(
                "hyperbolic_entailment_edges must be iterable"
            ) from error
        for index, edge in enumerate(raw_edges):
            if isinstance(edge, (str, bytes)):
                raise ValueError(
                    "hyperbolic_entailment_edges[{}] must be a pair".format(index)
                )
            try:
                parent, child = edge
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "hyperbolic_entailment_edges[{}] must contain exactly two "
                    "modalities".format(index)
                ) from error
            parent = cls._normalize_modality(
                parent,
                field="hyperbolic_entailment_edges[{}].parent".format(index),
            )
            child = cls._normalize_modality(
                child,
                field="hyperbolic_entailment_edges[{}].child".format(index),
            )
            if parent == child:
                raise ValueError(
                    "hyperbolic_entailment_edges[{}] is a self-edge".format(index)
                )
            pair = (parent, child)
            if pair in seen:
                raise ValueError(
                    "duplicate hyperbolic entailment edge {!r}".format(pair)
                )
            seen.add(pair)
            normalized.append(pair)
        return tuple(normalized)

    def _clamp_hyperbolic_parameters(self):
        with torch.no_grad():
            self.hyperbolic_log_curvature.clamp_(
                self._hyperbolic_log_curvature_min.item(),
                self._hyperbolic_log_curvature_max.item(),
            )
            for scale in self.hyperbolic_log_modality_scales.values():
                scale.clamp_(max=0.0)

    @property
    def hyperbolic_curvature(self):
        self._clamp_hyperbolic_parameters()
        return self.hyperbolic_log_curvature.exp()

    @staticmethod
    def _time_component(points, curvature):
        return torch.sqrt(
            (
                curvature.reciprocal()
                + points.square().sum(dim=-1, keepdim=True)
            ).clamp_min(torch.finfo(points.dtype).tiny)
        )

    def _pairwise_lorentz_inner(self, left, right):
        curvature = self.hyperbolic_curvature
        left_time = self._time_component(left, curvature)
        right_time = self._time_component(right, curvature)
        return (
            left @ right.transpose(-2, -1)
            - left_time @ right_time.transpose(-2, -1)
        )

    def _pairwise_distance(self, left, right):
        curvature = self.hyperbolic_curvature
        normalized_inner = (
            -curvature * self._pairwise_lorentz_inner(left, right)
        ).clamp_min(1.0 + 1e-6)
        return torch.acosh(normalized_inner) / curvature.sqrt()

    def _lift(self, features, modality):
        with autocast(enabled=False):
            features = F.normalize(features.float(), dim=-1)
            direction = F.normalize(
                self.hyperbolic_direction_heads[modality](features), dim=-1
            )
            log_radius_delta = math.log(3.0) * torch.tanh(
                self.hyperbolic_radius_heads[modality](features)
            )
            tangent = direction * torch.exp(
                self.hyperbolic_log_modality_scales[modality]
                + log_radius_delta
            )
            curvature = self.hyperbolic_curvature
            scaled_norm = curvature.sqrt() * torch.linalg.vector_norm(
                tangent, dim=-1, keepdim=True
            )
            sinh_input = scaled_norm.clamp(
                min=1e-6, max=math.asinh(2**15)
            )
            return (
                torch.sinh(sinh_input)
                * tangent
                / scaled_norm.clamp_min(1e-6)
            )

    def batch_get(self, batch, key):
        modality_by_key = {"feat_t": "t", "feat_v": "v", "feat_a": "a"}
        if key not in modality_by_key:
            return super().batch_get(batch, key)
        if key in batch:
            return batch[key]

        modality = modality_by_key[key]
        if modality == "t":
            output = super().batch_get(batch, "caption_output")
            pooled = self.pool_text_for_contra(output)
            euclidean = self.contra_head_t(pooled)
        elif modality == "v":
            output = super().batch_get(batch, "vision_output")
            pooled = self.pool_vision_for_contra(output)
            euclidean = self.contra_head_v(pooled)
        else:
            output = super().batch_get(batch, "audio_output")
            pooled = self.pool_audio_for_contra(output)
            euclidean = self.contra_head_a(pooled)
        batch[key] = self._lift(euclidean, modality)
        return batch[key]

    @staticmethod
    def _distributed_batch(values):
        if dist.is_available() and dist.is_initialized():
            gathered = all_gather_with_grad(values)
            rank = dist.get_rank()
        else:
            gathered = values
            rank = 0
        targets = (
            torch.arange(values.shape[0], device=values.device)
            + rank * values.shape[0]
        )
        return gathered, targets

    def _matched_reciprocal_similarity(self, left, right):
        curvature = self.hyperbolic_curvature
        left_time = self._time_component(left, curvature).squeeze(-1)
        right_time = self._time_component(right, curvature).squeeze(-1)
        inner = (left * right).sum(dim=-1) - left_time * right_time
        return (-curvature * inner).clamp_min(1.0).reciprocal()

    def _spectral_candidate_scores(self, text, vision, audio):
        similarity_tv = self._matched_reciprocal_similarity(text, vision)
        similarity_ta = self._matched_reciprocal_similarity(text, audio)
        similarity_va = self._matched_reciprocal_similarity(vision, audio)
        similarity_tv, similarity_ta, similarity_va = torch.broadcast_tensors(
            similarity_tv, similarity_ta, similarity_va
        )
        diagonal = torch.ones_like(similarity_tv)
        reciprocal_grams = torch.stack(
            (
                torch.stack((diagonal, similarity_tv, similarity_ta), dim=-1),
                torch.stack((similarity_tv, diagonal, similarity_va), dim=-1),
                torch.stack((similarity_ta, similarity_va, diagonal), dim=-1),
            ),
            dim=-2,
        )
        eigenvalues = torch.linalg.eigvalsh(reciprocal_grams)
        return eigenvalues[..., -1] - eigenvalues[..., -2]

    def _entailment_violation(self, parent, child):
        curvature = self.hyperbolic_curvature
        eps = 1e-6
        parent_time = self._time_component(parent, curvature).squeeze(-1)
        child_time = self._time_component(child, curvature).squeeze(-1)
        parent_norm = torch.linalg.vector_norm(parent, dim=-1)

        aperture_argument = (
            2.0 * self.hyperbolic_entailment_min_radius
            / (parent_norm * curvature.sqrt() + eps)
        )
        aperture = torch.asin(
            aperture_argument.clamp(min=-1.0 + eps, max=1.0 - eps)
        )

        curved_inner = curvature * (
            (parent * child).sum(dim=-1) - parent_time * child_time
        )
        angle_numerator = child_time + curved_inner * parent_time
        angle_denominator = parent_norm * torch.sqrt(
            (curved_inner.square() - 1.0).clamp_min(eps)
        ) + eps
        angle = torch.acos(
            (angle_numerator / angle_denominator).clamp(
                min=-1.0 + eps, max=1.0 - eps
            )
        )
        return F.relu(angle - aperture)

    def common_parent(self, text, vision, audio, return_details=False):
        """Mean or SVD proxy direction with a mean-based or learned residual radius.

        The cone is undefined at the origin. A small explicit domain margin
        keeps the constructed parent outside the cone's excluded inner ball.
        No detach; the learned-radius variant adds a zero-initialized scalar head.
        """
        with autocast(enabled=False):
            curvature = self.hyperbolic_curvature
            sqrt_c = curvature.sqrt()
            points = torch.stack((text.float(), vision.float(), audio.float()), dim=0)
            spatial_norm = torch.linalg.vector_norm(points, dim=-1, keepdim=True)
            radial_distance = torch.asinh(sqrt_c * spatial_norm) / sqrt_c
            origin_logs = points * (radial_distance / spatial_norm.clamp_min(1e-8))
            mean_log = origin_logs.mean(dim=0)
            mean_norm = torch.linalg.vector_norm(mean_log, dim=-1, keepdim=True)
            if bool((mean_norm <= 1e-8).any()):
                raise FloatingPointError("common parent direction is undefined: zero tangent mean")
            direction = mean_log / mean_norm
            raw_radius = self.hyperbolic_parent_contraction * mean_norm
            details = {}
            if self.hyperbolic_entailment_parent.startswith("svd_"):
                # exp_0 scales direction by a positive scalar, so normalization
                # recovers exactly the pre-exp direction-head output.
                directions = (points / spatial_norm.clamp_min(1e-8)).movedim(0, -2)
                direction = svd_proxy(directions)
                multiplier = torch.ones_like(raw_radius)
                if self.hyperbolic_entailment_parent == "svd_learned_radius":
                    multiplier = torch.exp(math.log(3.0) * torch.tanh(
                        self.hyperbolic_parent_radius_head(direction)))
                raw_radius = raw_radius * multiplier
                if return_details:
                    singular = torch.linalg.svdvals(directions)
                    cosines = torch.einsum('bmd,bd->bm', directions, direction)
                    details.update(proxy_sigma1=singular[:, 0],
                        proxy_gap=singular[:, 0].square()-singular[:, 1].square(),
                        radius_multiplier=multiplier.squeeze(-1))
                    for i, m in enumerate(self.modalities):
                        details['proxy_cosine_'+m] = cosines[:, i]
                        details['proxy_weight_'+m] = cosines[:, i]/singular[:, 0]
            floor = torch.asinh(torch.as_tensor(
                2 * self.hyperbolic_entailment_min_radius * (1 + self.hyperbolic_parent_floor_margin),
                device=points.device, dtype=points.dtype)) / sqrt_c
            radius = torch.maximum(raw_radius, floor)
            parent = direction * (torch.sinh(sqrt_c * radius) / sqrt_c)
            if return_details:
                return parent, dict(details, raw_radius=raw_radius.squeeze(-1), radius=radius.squeeze(-1),
                                    floor=floor, floor_active=(raw_radius < floor).squeeze(-1))
            return parent

    def entailment_loss(self, text, vision, audio):
        """Mean Lorentz-cone violation over the configured directed edges."""
        with autocast(enabled=False):
            features = {
                "t": text.float(),
                "v": vision.float(),
                "a": audio.float(),
            }
            if getattr(self, "hyperbolic_entailment_parent", "modality_edges") in ("tangent_mean", "svd_mean_radius", "svd_learned_radius"):
                parent = self.common_parent(features["t"], features["v"], features["a"])
                return torch.stack([self._entailment_violation(parent, features[m]).mean()
                                    for m in self.modalities]).mean()
            if not self.hyperbolic_entailment_edges:
                return sum(value.sum() for value in features.values()) * 0.0
            edge_losses = [
                self._entailment_violation(
                    features[parent], features[child]
                ).mean()
                for parent, child in self.hyperbolic_entailment_edges
            ]
            return torch.stack(edge_losses).mean()

    def spectral_loss(self, text, vision, audio):
        """Symmetric singleton-anchor vs paired-other-modalities loss."""
        with autocast(enabled=False):
            features = {
                "t": text.float(),
                "v": vision.float(),
                "a": audio.float(),
            }
            gathered = {}
            targets = None
            for modality in self.modalities:
                gathered[modality], modality_targets = self._distributed_batch(
                    features[modality]
                )
                if targets is None:
                    targets = modality_targets

            anchor = self.hyperbolic_spectral_anchor
            singleton_candidates = {
                modality: (
                    gathered[modality].unsqueeze(0)
                    if modality == anchor
                    else features[modality].unsqueeze(1)
                )
                for modality in self.modalities
            }
            paired_candidates = {
                modality: (
                    features[modality].unsqueeze(1)
                    if modality == anchor
                    else gathered[modality].unsqueeze(0)
                )
                for modality in self.modalities
            }
            singleton_candidate_scores = self._spectral_candidate_scores(
                singleton_candidates["t"],
                singleton_candidates["v"],
                singleton_candidates["a"],
            )
            paired_candidate_scores = self._spectral_candidate_scores(
                paired_candidates["t"],
                paired_candidates["v"],
                paired_candidates["a"],
            )
            return 0.5 * (
                F.cross_entropy(
                    singleton_candidate_scores
                    / self.hyperbolic_spectral_temperature,
                    targets,
                )
                + F.cross_entropy(
                    paired_candidate_scores
                    / self.hyperbolic_spectral_temperature,
                    targets,
                )
            )

    def retrieval_similarity(self, text, vision):
        with autocast(enabled=False):
            return -self._pairwise_distance(text.float(), vision.float())

    def _hard_negative_scores(self, text, condition):
        text_all = concat_all_gather(text)
        condition_all = concat_all_gather(condition)
        condition_to_text = -self._pairwise_distance(condition, text_all) / 0.07
        text_to_condition = -self._pairwise_distance(text, condition_all) / 0.07
        return condition_to_text, text_to_condition

    def forward_ret(self, batch, task, compute_loss=True):
        if not compute_loss:
            return super().forward_ret(batch, task, compute_loss=False)
        if self.ret_pairwise_stage1_train:
            # The base implementation dispatches back to this class's
            # Lorentz retrieval_similarity and hyperbolic batch_get methods.
            return super().forward_ret(batch, task, compute_loss=True)
        if isinstance(batch.raw_captions[0], list):
            batch.raw_captions = [
                caption for captions in batch.raw_captions for caption in captions
            ]

        subtasks = task.split("%")[1:]
        if any(subtask not in ("tv", "ta") for subtask in subtasks):
            raise ValueError("training supports only ret%tv%ta")

        feat_t = self.batch_get(batch, "feat_t")
        feat_v = self.batch_get(batch, "feat_v")
        feat_a = self.batch_get(batch, "feat_a")
        loss_dict = {
            "loss_hyperbolic_spectral": self.spectral_loss(
                feat_t, feat_v, feat_a
            )
        }
        if self.hyperbolic_entailment_weight > 0:
            loss_dict["loss_hyperbolic_entailment"] = (
                self.hyperbolic_entailment_weight
                * self.entailment_loss(feat_t, feat_v, feat_a)
            )

        caption_tokens = self.batch_get(batch, "caption_tokens")
        input_ids = caption_tokens.input_ids
        attention_mask = caption_tokens.attention_mask
        input_ids_collate = concat_all_gather(input_ids)
        attention_mask_collate = concat_all_gather(attention_mask)
        rank = dist.get_rank()
        batch_size = feat_t.shape[0]
        itm_losses = []

        for subtask in subtasks:
            condition = feat_v if subtask == "tv" else feat_a
            sim_cond2t, sim_t2cond = self._hard_negative_scores(
                feat_t, condition
            )
            condition_feats = self.batch_get(
                batch, "condition_feats_{}".format(subtask[1:])
            )
            condition_feats_collate = all_gather_with_grad(condition_feats)
            with torch.no_grad():
                weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                weights_t2cond[
                    :, rank * batch_size : rank * batch_size + batch_size
                ].fill_diagonal_(0)
                weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                weights_cond2t[
                    :, rank * batch_size : rank * batch_size + batch_size
                ].fill_diagonal_(0)

            condition_feats_neg = torch.stack(
                [
                    condition_feats_collate[
                        torch.multinomial(weights_t2cond[index], 1).item()
                    ]
                    for index in range(batch_size)
                ],
                dim=0,
            )
            negative_text_indices = [
                torch.multinomial(weights_cond2t[index], 1).item()
                for index in range(batch_size)
            ]
            text_ids_neg = torch.stack(
                [input_ids_collate[index] for index in negative_text_indices], dim=0
            )
            text_atts_neg = torch.stack(
                [
                    attention_mask_collate[index]
                    for index in negative_text_indices
                ],
                dim=0,
            )

            itm_input_ids = torch.cat(
                (input_ids, input_ids, text_ids_neg), dim=0
            )
            itm_attention_mask = torch.cat(
                (attention_mask, attention_mask, text_atts_neg), dim=0
            )
            itm_condition_feats = torch.cat(
                (condition_feats, condition_feats_neg, condition_feats), dim=0
            )
            output = self.multimodal_encoder.bert(
                input_ids=itm_input_ids,
                attention_mask=itm_attention_mask,
                encoder_hidden_states=itm_condition_feats,
            ).last_hidden_state
            logits = self.itm_head(output[:, 0].half())
            ground_truth = torch.zeros(
                batch_size * 3, dtype=torch.long, device=logits.device
            )
            ground_truth[:batch_size] = 1
            itm_losses.append(F.cross_entropy(logits, ground_truth))

        loss_dict["loss_itm"] = (
            self.lambda_itm * sum(itm_losses) / len(itm_losses)
        )
        return loss_dict
