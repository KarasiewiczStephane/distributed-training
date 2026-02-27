"""DataLoader factory with optimized defaults and wrapper class."""

import logging
from typing import Iterator, Optional

from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """Create a DataLoader with performance-tuned defaults.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle (ignored when sampler is provided).
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        sampler: Optional sampler (overrides shuffle).

    Returns:
        Configured DataLoader instance.
    """
    effective_shuffle = shuffle if sampler is None else False
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
    )
    logger.info(
        "Created DataLoader (batch_size=%d, num_workers=%d, pin_memory=%s)",
        batch_size,
        num_workers,
        pin_memory,
    )
    return loader


class OptimizedDataLoader:
    """Wrapper around DataLoader with optimized defaults for distributed training.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Whether to keep workers alive between iterations.
        sampler: Optional sampler for distributed training.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        sampler: Optional[Sampler] = None,
    ) -> None:
        effective_persistent = persistent_workers and num_workers > 0
        prefetch = 2 if num_workers > 0 else None

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=effective_persistent,
            prefetch_factor=prefetch,
            drop_last=True,
        )

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.loader)
