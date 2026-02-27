"""Integration tests for end-to-end training workflows."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import SingleGPUTrainer
from src.utils.config import load_config


def _make_synthetic_loader(n=128, batch_size=32, in_features=3 * 4 * 4):
    """Create a synthetic DataLoader."""
    x = torch.randn(n, in_features)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, drop_last=True)


def _make_model(in_features=48, num_classes=10):
    """Create a simple model for integration tests."""
    return nn.Sequential(
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


class TestSingleEpochTraining:
    """Test that a single epoch of training works end-to-end."""

    def test_loss_decreases_over_epoch(self, training_config_path):
        """Training loss should decrease across an epoch."""
        config = load_config(str(training_config_path))
        model = _make_model()
        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n=64)

        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
        result = trainer.train()

        assert len(result["history"]) == config.training.epochs
        first_loss = result["history"][0]["loss"]
        last_loss = result["history"][-1]["loss"]
        assert last_loss <= first_loss

    def test_train_returns_history(self, training_config_path):
        """Training should return a dict with a history list."""
        config = load_config(str(training_config_path))
        model = _make_model()
        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n=64)

        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
        result = trainer.train()

        assert "history" in result
        for entry in result["history"]:
            assert "loss" in entry
            assert "accuracy" in entry
            assert "val_loss" in entry
            assert "val_accuracy" in entry

    def test_model_parameters_change(self, training_config_path):
        """Model parameters should change after training."""
        config = load_config(str(training_config_path))
        model = _make_model()
        initial_params = [p.clone() for p in model.parameters()]

        train_loader = _make_synthetic_loader()
        val_loader = _make_synthetic_loader(n=64)

        trainer = SingleGPUTrainer(model, train_loader, val_loader, config)
        trainer.train()

        changed = False
        for old, new in zip(initial_params, model.parameters()):
            if not torch.equal(old, new):
                changed = True
                break
        assert changed
