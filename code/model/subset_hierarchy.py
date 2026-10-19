"""VAST feature extraction plus the independently testable seven-subset heads."""
from .pmrl import PMRL
from utils.subset_hierarchy import SubsetHeads


class SubsetHierarchy(PMRL):
    def __init__(self, config):
        super().__init__(config)
        self.subset_heads = SubsetHeads(
            {"t": self.multimodal_dim, "v": self.vision_dim, "a": self.audio_dim},
            config.contra_dim, getattr(config, "subset_geometry", "hyperbolic"))

    def pooled_features(self, batch):
        return {
            "t": self.pool_text_for_contra(self.batch_get(batch, "caption_output")),
            "v": self.pool_vision_for_contra(self.batch_get(batch, "vision_output")),
            "a": self.pool_audio_for_contra(self.batch_get(batch, "audio_output")),
        }

    def configure_joint(self):
        self.requires_grad_(False)
        # BEATs has stochastic LayerDrop. Non-reentrant checkpointing permits
        # DDP unused-parameter detection without assuming a static layer graph.
        for module in self.modules():
            module.subset_non_reentrant = True
        for module in (self.vision_encoder, self.audio_encoder, self.multimodal_encoder.bert, self.subset_heads):
            module.requires_grad_(True)

    def forward(self, batch, task=None, compute_loss=True):
        # The dedicated epoch runner owns the objective and validation protocol.
        if task is not None:
            raise ValueError('Use experiments/subset_hierarchy/train.py for subset training/evaluation')
        return self.subset_heads(self.pooled_features(batch))
