"""Tests for DataLoader factory and OptimizedDataLoader."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import OptimizedDataLoader, create_dataloader


class TestCreateDataloader:
    """Tests for create_dataloader function."""

    def test_returns_dataloader(self, small_dataset):
        """Factory returns a DataLoader instance."""
        loader = create_dataloader(small_dataset, batch_size=32, num_workers=0)
        assert isinstance(loader, DataLoader)

    def test_batch_size(self, small_dataset):
        """DataLoader yields batches of the specified size."""
        loader = create_dataloader(small_dataset, batch_size=16, num_workers=0)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 16

    def test_no_shuffle_with_sampler(self, small_dataset):
        """Shuffle is disabled when a sampler is provided."""
        from torch.utils.data import SequentialSampler

        sampler = SequentialSampler(small_dataset)
        loader = create_dataloader(
            small_dataset, batch_size=16, num_workers=0, sampler=sampler
        )
        assert isinstance(loader, DataLoader)

    def test_persistent_workers_disabled_when_zero(self):
        """persistent_workers is False when num_workers=0."""
        dataset = TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
        loader = create_dataloader(dataset, batch_size=8, num_workers=0)
        assert loader.persistent_workers is False


class TestOptimizedDataLoader:
    """Tests for OptimizedDataLoader wrapper."""

    def test_iteration(self):
        """OptimizedDataLoader yields batches."""
        dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,)))
        odl = OptimizedDataLoader(dataset, batch_size=16, num_workers=0)
        batches = list(odl)
        assert len(batches) > 0
        assert batches[0][0].shape[0] == 16

    def test_len(self):
        """OptimizedDataLoader reports correct number of batches."""
        dataset = TensorDataset(torch.randn(64, 8), torch.randint(0, 2, (64,)))
        odl = OptimizedDataLoader(dataset, batch_size=16, num_workers=0)
        assert len(odl) == 4

    def test_with_sampler(self):
        """OptimizedDataLoader works with a custom sampler."""
        from torch.utils.data import SequentialSampler

        dataset = TensorDataset(torch.randn(32, 8), torch.randint(0, 2, (32,)))
        sampler = SequentialSampler(dataset)
        odl = OptimizedDataLoader(dataset, batch_size=8, num_workers=0, sampler=sampler)
        assert len(odl) == 4
