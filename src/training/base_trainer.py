"""Base trainer with single-GPU training loop, LR scheduling, and gradient clipping."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer providing a standard single-GPU training loop.

    Subclasses must implement the ``train`` method to define the full
    training procedure (e.g. multi-epoch loop with validation).

    Args:
        model: The neural network to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        config: Training configuration (OmegaConf DictConfig).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self._setup_scheduler()
        logger.info("BaseTrainer initialized on device=%s", self.device)

    def _setup_scheduler(self) -> None:
        """Set up learning rate warmup followed by cosine annealing."""
        warmup_epochs = self.config.training.warmup_epochs
        total_epochs = self.config.training.epochs

        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dictionary with 'loss' and 'accuracy' for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip,
            )
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation on the validation set.

        Returns:
            Dictionary with 'val_loss' and 'val_accuracy'.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0.0
        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Run the full training procedure.

        Returns:
            Dictionary with final training metrics.
        """


class SingleGPUTrainer(BaseTrainer):
    """Concrete trainer for single-GPU (non-distributed) training.

    Runs the full training loop with validation, LR scheduling,
    and metric logging.
    """

    def train(self) -> dict[str, Any]:
        """Run training for the configured number of epochs.

        Returns:
            Dictionary containing training history.
        """
        history: list[dict[str, float]] = []
        num_epochs = self.config.training.epochs

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(metrics)
            logger.info(
                "Epoch %d/%d - loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f",
                epoch + 1,
                num_epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
            )

        return {"history": history}
