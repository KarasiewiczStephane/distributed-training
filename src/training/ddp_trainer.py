"""PyTorch DistributedDataParallel trainer with torchrun support."""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class DDPTrainer:
    """Distributed Data Parallel trainer using PyTorch DDP.

    Manages multi-process training with NCCL/gloo backends, DistributedSampler
    for data partitioning, and automatic gradient synchronization.

    Args:
        model: The neural network to train (unwrapped).
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self._init_distributed()

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        if dist.is_initialized() and self.world_size > 1:
            device_ids = [self.local_rank] if torch.cuda.is_available() else None
            self.model = DDP(self.model, device_ids=device_ids)

        per_gpu_batch = config.training.batch_size // self.world_size

        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.world_size, rank=self.rank
        )
        self.val_sampler = DistributedSampler(
            val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_gpu_batch,
            sampler=self.train_sampler,
            num_workers=config.data.get("num_workers", 4),
            pin_memory=config.data.get("pin_memory", True),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch,
            sampler=self.val_sampler,
            num_workers=config.data.get("num_workers", 4),
            pin_memory=config.data.get("pin_memory", True),
        )

        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self._setup_scheduler()

        logger.info(
            "DDPTrainer initialized (rank=%d, world_size=%d, device=%s)",
            self.rank,
            self.world_size,
            self.device,
        )

    def _init_distributed(self) -> None:
        """Initialize the distributed process group if not already done."""
        if not dist.is_initialized() and self.world_size > 1:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            logger.info("Initialized process group with backend=%s", backend)

    def _setup_scheduler(self) -> None:
        """Set up learning rate warmup followed by cosine annealing."""
        warmup_epochs = self.config.training.warmup_epochs
        total_epochs = self.config.training.epochs
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process.

        Returns:
            True if rank 0, False otherwise.
        """
        return self.rank == 0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one distributed training epoch.

        Args:
            epoch: Current epoch number (used by DistributedSampler for shuffling).

        Returns:
            Dictionary with 'loss' and 'accuracy' for the epoch.
        """
        self.train_sampler.set_epoch(epoch)
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
                self.model.parameters(), self.config.training.gradient_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation across all processes.

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

        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0
        return {"val_loss": avg_loss, "val_accuracy": accuracy}

    def train(self) -> dict[str, Any]:
        """Run the full DDP training procedure.

        Returns:
            Dictionary containing training history.
        """
        history: list[dict[str, float]] = []
        num_epochs = self.config.training.epochs

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(metrics)

            if self.is_main_process():
                logger.info(
                    "Epoch %d/%d - loss: %.4f, acc: %.4f, "
                    "val_loss: %.4f, val_acc: %.4f",
                    epoch + 1,
                    num_epochs,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    val_metrics["val_loss"],
                    val_metrics["val_accuracy"],
                )

        return {"history": history}

    def cleanup(self) -> None:
        """Destroy the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed process group")
