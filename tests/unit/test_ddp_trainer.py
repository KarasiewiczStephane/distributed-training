"""Tests for the DDP trainer module."""

from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.training.ddp_trainer import DDPTrainer
from src.utils.config import load_config


def _make_tiny_dataset(n_samples: int = 64):
    """Create a small synthetic dataset."""
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (n_samples,))
    return TensorDataset(images, labels)


class TestDDPTrainerInit:
    """Tests for DDPTrainer initialization."""

    def test_single_process_init(self, training_config_path):
        """DDPTrainer initializes in single-process mode (world_size=1)."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert trainer.is_main_process()

    def test_batch_size_scaling(self, training_config_path):
        """Batch size is divided by world_size for per-GPU batches."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        expected_batch = config.training.batch_size // 1
        assert trainer.train_loader.batch_size == expected_batch

    def test_distributed_sampler_created(self, training_config_path):
        """DistributedSampler is created for training data."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        from torch.utils.data.distributed import DistributedSampler

        assert isinstance(trainer.train_sampler, DistributedSampler)


class TestDDPTrainerTraining:
    """Tests for DDP training loop."""

    def test_train_epoch_returns_metrics(self, training_config_path):
        """train_epoch returns loss and accuracy."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        metrics = trainer.train_epoch(epoch=0)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0

    def test_validate_returns_metrics(self, training_config_path):
        """validate returns val_loss and val_accuracy."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        val_metrics = trainer.validate()
        assert "val_loss" in val_metrics
        assert "val_accuracy" in val_metrics

    def test_full_training_loop(self, training_config_path):
        """Full training runs for configured epochs."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        result = trainer.train()
        assert len(result["history"]) == config.training.epochs

    def test_is_main_process(self, training_config_path):
        """is_main_process returns True for rank 0."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        dataset = _make_tiny_dataset()

        env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
        with patch.dict("os.environ", env, clear=False):
            trainer = DDPTrainer(model, dataset, dataset, config)

        assert trainer.is_main_process() is True
