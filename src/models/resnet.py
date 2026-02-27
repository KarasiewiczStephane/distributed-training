"""ResNet-50 model factory for CIFAR-100 classification."""

import logging

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

logger = logging.getLogger(__name__)


def create_resnet50(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """Create a ResNet-50 model adapted for the given number of classes.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        ResNet-50 model with adjusted final fully-connected layer.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    logger.info(
        "Created ResNet-50 (num_classes=%d, pretrained=%s)", num_classes, pretrained
    )
    return model
