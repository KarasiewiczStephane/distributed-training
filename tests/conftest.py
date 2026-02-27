"""Shared test fixtures for the distributed training test suite."""

import pytest
import torch
from torch.utils.data import TensorDataset


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for CI-compatible testing."""
    return torch.device("cpu")


@pytest.fixture
def small_dataset() -> TensorDataset:
    """Create a small synthetic dataset for fast testing.

    Returns:
        TensorDataset with 256 random 3x32x32 images and 100-class labels.
    """
    images = torch.randn(256, 3, 32, 32)
    labels = torch.randint(0, 100, (256,))
    return TensorDataset(images, labels)


@pytest.fixture
def training_config_path(tmp_path) -> str:
    """Write a minimal training config to a temp file and return its path."""
    config_content = """
model:
  name: resnet50
  num_classes: 100
  pretrained: false
training:
  epochs: 2
  batch_size: 32
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  warmup_epochs: 1
  gradient_clip: 1.0
data:
  root: ./data
  dataset: cifar100
  num_workers: 0
  pin_memory: false
checkpoint:
  save_dir: {save_dir}
  save_every: 1
  early_stopping_patience: 3
wandb:
  project: test
  enabled: false
""".format(save_dir=str(tmp_path / "checkpoints"))
    config_file = tmp_path / "training.yaml"
    config_file.write_text(config_content)
    return str(config_file)
