"""Tests for ResNet-50 model creation."""

import torch

from src.models.resnet import create_resnet50


class TestCreateResnet50:
    """Tests for create_resnet50 factory function."""

    def test_default_output_shape(self):
        """Model output shape matches (batch, num_classes) with default 100 classes."""
        model = create_resnet50(num_classes=100, pretrained=False)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 100)

    def test_custom_num_classes(self):
        """Model adapts to a custom number of output classes."""
        model = create_resnet50(num_classes=10, pretrained=False)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_model_has_parameters(self):
        """Model has trainable parameters."""
        model = create_resnet50(num_classes=100, pretrained=False)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_model_is_nn_module(self):
        """Returned object is an nn.Module."""
        model = create_resnet50()
        assert isinstance(model, torch.nn.Module)
