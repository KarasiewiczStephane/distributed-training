"""CIFAR-100 dataset loading with standard augmentation transforms."""

import logging
from typing import Optional

from torch.utils.data import Dataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_transforms(train: bool = True) -> transforms.Compose:
    """Build standard CIFAR-100 transforms.

    Training transforms include random crop and horizontal flip.
    Evaluation transforms only normalize.

    Args:
        train: Whether to include training augmentations.

    Returns:
        Composed torchvision transforms.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def get_cifar100(
    root: str,
    train: bool,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
) -> Dataset:
    """Load the CIFAR-100 dataset.

    Args:
        root: Root directory for dataset storage.
        train: Whether to load the training split.
        download: Whether to download if not present locally.
        transform: Optional override for transforms. If None, uses defaults.

    Returns:
        CIFAR-100 dataset instance.
    """
    if transform is None:
        transform = get_cifar100_transforms(train=train)
    logger.info("Loading CIFAR-100 (train=%s) from %s", train, root)
    return datasets.CIFAR100(
        root=root, train=train, download=download, transform=transform
    )
