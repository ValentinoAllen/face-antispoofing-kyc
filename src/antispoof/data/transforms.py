"""Image transforms.

The baseline transform has no augmentation and no face crop: the bounding-box format is still an
open question (``docs/SCHEMA.md`` §1.4).
"""

from timm.data import resolve_model_data_config
from torch import nn
from torchvision import transforms

from antispoof.data.dataset import ImageTransform


def build_baseline_transform(input_size: int, model: nn.Module) -> ImageTransform:
    """Resize to a square, convert to a tensor and normalize with the backbone's pretrained stats.

    Mean, standard deviation and resize interpolation are read from the model's timm pretrained
    config (``timm.data.resolve_model_data_config``), so they follow the configured backbone.

    Args:
        input_size: Side length of the square output in pixels (``model.input_size``).
        model: A timm model. Only its pretrained config is read.

    Returns:
        A callable mapping an RGB PIL image to a float tensor of shape
        ``(3, input_size, input_size)``.

    Raises:
        ValueError: If ``input_size`` is not positive.
    """
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}.")
    data_config = resolve_model_data_config(model)
    transform: ImageTransform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode(data_config["interpolation"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ]
    )
    return transform
