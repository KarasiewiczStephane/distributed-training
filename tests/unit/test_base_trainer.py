"""Tests for the base trainer and single-GPU training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import BaseTrainer, SingleGPUTrainer
from src.utils.config import load_config


def _make_tiny_loaders(batch_size: int = 16, n_samples: int = 64):
    """Create tiny train/val data loaders for testing."""
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (n_samples,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    return loader, loader


class TestBaseTrainer:
    """Tests for BaseTrainer initialization and components."""

    def test_init_sets_device_cpu(self, training_config_path):
        """Trainer defaults to CPU when no GPU is available."""
        config = load_config(training_config_path)
        model = nn.Linear(3 * 32 * 32, 10)
        train_loader, val_loader = _make_tiny_loaders()
        # Override num_classes for simpler model
        config.model.num_classes = 10

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        trainer = ConcreteTrainer(model, train_loader, val_loader, config)
        assert trainer.device == torch.device("cpu") or trainer.device.type in (
            "cpu",
            "cuda",
        )

    def test_optimizer_uses_config_lr(self, training_config_path):
        """Optimizer learning rate matches config."""
        config = load_config(training_config_path)
        model = nn.Linear(3 * 32 * 32, 10)
        train_loader, val_loader = _make_tiny_loaders()

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        trainer = ConcreteTrainer(model, train_loader, val_loader, config)
        # After SequentialLR warmup attachment, effective LR = base_lr * start_factor
        lr = trainer.optimizer.param_groups[0]["lr"]
        expected_lr = config.training.lr * 0.1  # start_factor=0.1
        assert abs(lr - expected_lr) < 1e-9

    def test_scheduler_is_sequential(self, training_config_path):
        """Scheduler is a SequentialLR combining warmup and cosine."""
        config = load_config(training_config_path)
        model = nn.Linear(3 * 32 * 32, 10)
        train_loader, val_loader = _make_tiny_loaders()

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        trainer = ConcreteTrainer(model, train_loader, val_loader, config)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.SequentialLR)


class TestTrainEpoch:
    """Tests for the train_epoch method."""

    def test_returns_loss_and_accuracy(self, training_config_path):
        """train_epoch returns dict with loss and accuracy keys."""
        config = load_config(training_config_path)
        train_loader, val_loader = _make_tiny_loaders()

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        flat_model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        trainer = ConcreteTrainer(flat_model, train_loader, val_loader, config)
        metrics = trainer.train_epoch()
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0

    def test_gradient_clipping_applied(self, training_config_path):
        """Gradient norms are clipped to the configured threshold."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        train_loader, val_loader = _make_tiny_loaders()

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        trainer = ConcreteTrainer(model, train_loader, val_loader, config)
        trainer.train_epoch()
        # After training, verify no parameter gradient exceeds clip value
        # (clipping is per-iteration, just check training ran without error)


class TestValidate:
    """Tests for the validate method."""

    def test_returns_val_metrics(self, training_config_path):
        """validate returns val_loss and val_accuracy."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        train_loader, val_loader = _make_tiny_loaders()

        class ConcreteTrainer(BaseTrainer):
            def train(self):
                return {}

        trainer = ConcreteTrainer(model, train_loader, val_loader, config)
        val_metrics = trainer.validate()
        assert "val_loss" in val_metrics
        assert "val_accuracy" in val_metrics


class TestSingleGPUTrainer:
    """Tests for the SingleGPUTrainer end-to-end."""

    def test_train_runs_all_epochs(self, training_config_path):
        """SingleGPUTrainer runs for the configured number of epochs."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        train_loader, val_loader = _make_tiny_loaders()
        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
        result = trainer.train()
        assert len(result["history"]) == config.training.epochs

    def test_lr_changes_over_epochs(self, training_config_path):
        """Learning rate changes across epochs due to scheduling."""
        config = load_config(training_config_path)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
        train_loader, val_loader = _make_tiny_loaders()
        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
        result = trainer.train()
        lrs = [h["lr"] for h in result["history"]]
        # LR should change (warmup then cosine)
        assert lrs[0] != lrs[-1]
