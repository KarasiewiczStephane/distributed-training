"""Tests for CIFAR-100 dataset and transforms."""

import pytest
import torch
from torchvision import transforms

from src.data.dataset import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    get_cifar100_transforms,
)


class TestCifar100Transforms:
    """Tests for CIFAR-100 transform pipelines."""

    def test_train_transforms_include_augmentation(self):
        """Training transforms include RandomCrop and RandomHorizontalFlip."""
        t = get_cifar100_transforms(train=True)
        transform_types = [type(tr) for tr in t.transforms]
        assert transforms.RandomCrop in transform_types
        assert transforms.RandomHorizontalFlip in transform_types

    def test_eval_transforms_no_augmentation(self):
        """Evaluation transforms do not include data augmentation."""
        t = get_cifar100_transforms(train=False)
        transform_types = [type(tr) for tr in t.transforms]
        assert transforms.RandomCrop not in transform_types
        assert transforms.RandomHorizontalFlip not in transform_types

    def test_both_include_normalization(self):
        """Both train and eval transforms normalize with CIFAR-100 stats."""
        for train in [True, False]:
            t = get_cifar100_transforms(train=train)
            normalize = [
                tr for tr in t.transforms if isinstance(tr, transforms.Normalize)
            ]
            assert len(normalize) == 1
            assert tuple(normalize[0].mean) == pytest.approx(CIFAR100_MEAN)
            assert tuple(normalize[0].std) == pytest.approx(CIFAR100_STD)

    def test_output_shape(self):
        """Transforms produce a (3, 32, 32) tensor from a PIL-like input."""
        t = get_cifar100_transforms(train=True)
        from PIL import Image

        img = Image.new("RGB", (32, 32))
        result = t(img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 32, 32)
