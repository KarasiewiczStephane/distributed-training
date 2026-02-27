"""Tests for DataLoader factory."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import create_dataloader


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
