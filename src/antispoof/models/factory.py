"""Build classification models from the ``model:`` section of an experiment config."""

import timm
from torch import nn

from antispoof.training.config import ModelConfig

NUM_LOGITS = 1
"""One logit per image. ``sigmoid(logit)`` is P(spoof), matching label index 43 (1 = spoof)."""


def build_model(config: ModelConfig) -> nn.Module:
    """Create a timm backbone with a single-logit classification head.

    Args:
        config: Backbone name and whether to load its pretrained weights.

    Returns:
        The model. Its output has shape ``(batch, 1)``.
    """
    model: nn.Module = timm.create_model(
        config.backbone, pretrained=config.pretrained, num_classes=NUM_LOGITS
    )
    return model
