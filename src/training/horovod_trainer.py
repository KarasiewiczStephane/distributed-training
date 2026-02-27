"""Horovod-based distributed training as an alternative to PyTorch DDP."""

import logging
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

try:
    import horovod.torch as hvd

    HOROVOD_AVAILABLE = True
except ImportError:
    hvd = None
    HOROVOD_AVAILABLE = False


class HorovodTrainer:
    """Distributed trainer using Horovod for gradient synchronization.

    Uses hvd.init(), DistributedOptimizer, and broadcast_parameters
    for multi-worker training.

    Args:
        model: The neural network to train.
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
        if not HOROVOD_AVAILABLE:
            raise ImportError(
                "Horovod is not installed. Install with: pip install horovod"
            )

        self.config = config
        hvd.init()
        self.rank = hvd.rank()
        self.world_size = hvd.size()
        self.local_rank = hvd.local_rank()

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        # Scale learning rate by world size
        scaled_lr = config.training.lr * self.world_size
        self.optimizer = SGD(
            self.model.parameters(),
            lr=scaled_lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
        )

        # Wrap optimizer with Horovod distributed optimizer
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.model.named_parameters(),
            op=hvd.Average,
        )

        # Broadcast initial parameters and optimizer state
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        self.criterion = nn.CrossEntropyLoss()
        self._setup_scheduler()

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

        logger.info(
            "HorovodTrainer initialized (rank=%d, world_size=%d, device=%s)",
            self.rank,
            self.world_size,
            self.device,
        )

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
        """Check if this is the main (rank 0) process."""
        return self.rank == 0

    def _average_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Average metrics across all workers using Horovod allreduce.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Dictionary of averaged metrics.
        """
        averaged = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            averaged[key] = hvd.allreduce(tensor, op=hvd.Average).item()
        return averaged

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one Horovod training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with 'loss' and 'accuracy'.
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
            self.optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        n_batches = len(self.train_loader)
        metrics = {
            "loss": total_loss / n_batches if n_batches > 0 else 0,
            "accuracy": correct / total if total > 0 else 0,
        }
        return self._average_metrics(metrics)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and average results across workers.

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
        metrics = {
            "val_loss": total_loss / n_batches if n_batches > 0 else 0,
            "val_accuracy": correct / total if total > 0 else 0,
        }
        return self._average_metrics(metrics)

    def train(self) -> dict[str, Any]:
        """Run the full Horovod training procedure.

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
