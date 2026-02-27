"""Checkpoint management with rank-aware saving and distributed early stopping."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoint saving and loading.

    Only rank 0 writes to disk in distributed settings to avoid file conflicts.

    Args:
        save_dir: Directory to store checkpoint files.
        rank: Process rank (only rank 0 saves).
    """

    def __init__(self, save_dir: str, rank: int = 0) -> None:
        self.save_dir = Path(save_dir)
        self.rank = rank
        if self.rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict[str, Any], filename: str, is_best: bool = False) -> None:
        """Save a checkpoint to disk (rank 0 only).

        Args:
            state: State dictionary containing model, optimizer, epoch, etc.
            filename: Checkpoint filename (e.g. 'epoch_10.pt').
            is_best: Whether to also save as 'best.pt'.
        """
        if self.rank != 0:
            return
        path = self.save_dir / filename
        torch.save(state, path)
        logger.info("Saved checkpoint to %s", path)
        if is_best:
            best_path = self.save_dir / "best.pt"
            torch.save(state, best_path)
            logger.info("Saved best checkpoint to %s", best_path)

    def load(
        self, filename: str, map_location: str = "cpu"
    ) -> Optional[dict[str, Any]]:
        """Load a checkpoint from disk.

        Args:
            filename: Checkpoint filename to load.
            map_location: Device mapping for torch.load.

        Returns:
            State dictionary, or None if file does not exist.
        """
        path = self.save_dir / filename
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return None
        state = torch.load(path, map_location=map_location, weights_only=False)
        logger.info("Loaded checkpoint from %s", path)
        return state


class DistributedEarlyStopping:
    """Early stopping with broadcast coordination across distributed processes.

    Rank 0 decides whether to stop based on validation loss improvement,
    then broadcasts the decision to all other ranks.

    Args:
        patience: Number of epochs without improvement before stopping.
        rank: Process rank.
        world_size: Total number of processes.
        min_delta: Minimum loss decrease to count as improvement.
    """

    def __init__(
        self,
        patience: int,
        rank: int = 0,
        world_size: int = 1,
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.rank = rank
        self.world_size = world_size
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop_flag = False

    def should_stop(self, val_loss: float) -> bool:
        """Determine whether training should stop.

        Rank 0 evaluates the stopping criterion and broadcasts the result.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        device = "cpu"
        if torch.cuda.is_available() and dist.is_initialized():
            device = f"cuda:{self.rank}"

        stop = torch.tensor([0], dtype=torch.int, device=device)

        if self.rank == 0:
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                logger.info(
                    "Early stopping counter: %d/%d", self.counter, self.patience
                )
            if self.counter >= self.patience:
                stop[0] = 1

        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast(stop, src=0)

        self.should_stop_flag = stop.item() == 1
        return self.should_stop_flag
