"""Tests for Horovod trainer module (with mocked horovod)."""

import importlib
import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.utils.config import load_config


def _make_tiny_dataset(n_samples: int = 64):
    """Create a small synthetic dataset."""
    images = torch.randn(n_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (n_samples,))
    return TensorDataset(images, labels)


def _get_trainer_with_mock(world_size: int = 1):
    """Set up mocked horovod and return (mock_hvd, HorovodTrainer class).

    Args:
        world_size: Simulated number of workers.

    Returns:
        Tuple of (mock_hvd_module, HorovodTrainer_class).
    """
    mock_hvd = MagicMock()
    mock_hvd.init = MagicMock()
    mock_hvd.rank = MagicMock(return_value=0)
    mock_hvd.size = MagicMock(return_value=world_size)
    mock_hvd.local_rank = MagicMock(return_value=0)
    mock_hvd.Average = "average"
    mock_hvd.broadcast_parameters = MagicMock()
    mock_hvd.broadcast_optimizer_state = MagicMock()

    def distributed_optimizer(optimizer, named_parameters, op):
        return optimizer

    mock_hvd.DistributedOptimizer = distributed_optimizer

    def allreduce(tensor, op=None):
        return tensor

    mock_hvd.allreduce = allreduce

    horovod_mock = MagicMock()
    horovod_mock.torch = mock_hvd
    sys.modules["horovod"] = horovod_mock
    sys.modules["horovod.torch"] = mock_hvd

    import src.training.horovod_trainer as mod

    importlib.reload(mod)
    return mock_hvd, mod.HorovodTrainer


def _cleanup_mock():
    """Remove mocked horovod from sys.modules."""
    sys.modules.pop("horovod", None)
    sys.modules.pop("horovod.torch", None)
    import src.training.horovod_trainer as mod

    importlib.reload(mod)


class TestHorovodTrainerInit:
    """Tests for HorovodTrainer initialization."""

    def test_init_calls_hvd_init(self, training_config_path):
        """Horovod init is called during trainer creation."""
        mock_hvd, HorovodTrainer = _get_trainer_with_mock()
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            HorovodTrainer(model, dataset, dataset, config)
            mock_hvd.init.assert_called_once()
        finally:
            _cleanup_mock()

    def test_broadcast_called(self, training_config_path):
        """Parameters and optimizer state are broadcast from rank 0."""
        mock_hvd, HorovodTrainer = _get_trainer_with_mock()
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            HorovodTrainer(model, dataset, dataset, config)
            mock_hvd.broadcast_parameters.assert_called_once()
            mock_hvd.broadcast_optimizer_state.assert_called_once()
        finally:
            _cleanup_mock()

    def test_lr_scaled_by_world_size(self, training_config_path):
        """Learning rate is scaled by the number of workers."""
        _, HorovodTrainer = _get_trainer_with_mock(world_size=2)
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            trainer = HorovodTrainer(model, dataset, dataset, config)
            # After scheduler attachment, LR = scaled_lr * start_factor
            expected = config.training.lr * 2 * 0.1
            actual = trainer.optimizer.param_groups[0]["lr"]
            assert abs(actual - expected) < 1e-9
        finally:
            _cleanup_mock()


class TestHorovodTrainerTraining:
    """Tests for Horovod training loop."""

    def test_train_epoch_returns_metrics(self, training_config_path):
        """train_epoch returns loss and accuracy."""
        _, HorovodTrainer = _get_trainer_with_mock()
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            trainer = HorovodTrainer(model, dataset, dataset, config)
            metrics = trainer.train_epoch(epoch=0)
            assert "loss" in metrics
            assert "accuracy" in metrics
        finally:
            _cleanup_mock()

    def test_full_training_loop(self, training_config_path):
        """Full training runs for configured epochs."""
        _, HorovodTrainer = _get_trainer_with_mock()
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            trainer = HorovodTrainer(model, dataset, dataset, config)
            result = trainer.train()
            assert len(result["history"]) == config.training.epochs
        finally:
            _cleanup_mock()

    def test_is_main_process(self, training_config_path):
        """is_main_process returns True for rank 0."""
        _, HorovodTrainer = _get_trainer_with_mock()
        try:
            config = load_config(training_config_path)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
            dataset = _make_tiny_dataset()
            trainer = HorovodTrainer(model, dataset, dataset, config)
            assert trainer.is_main_process() is True
        finally:
            _cleanup_mock()
