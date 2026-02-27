"""Weights & Biases experiment tracking with rank-aware logging."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class WandBLogger:
    """Weights & Biases logger for distributed training experiments.

    Only the main process (rank 0) logs metrics to avoid duplicate entries.

    Args:
        project: W&B project name.
        config: Hyperparameter configuration to log.
        run_name: Optional name for the run.
        rank: Process rank (only rank 0 logs).
        enabled: Whether to enable W&B logging.
    """

    def __init__(
        self,
        project: str,
        config: dict[str, Any],
        run_name: Optional[str] = None,
        rank: int = 0,
        enabled: bool = True,
    ) -> None:
        self.rank = rank
        self.enabled = enabled and rank == 0 and WANDB_AVAILABLE
        self.run = None

        if self.enabled:
            self.run = wandb.init(
                project=project,
                config=config,
                name=run_name,
                settings=wandb.Settings(silent=True),
            )
            logger.info("W&B run initialized: %s", project)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training metrics to W&B.

        Args:
            metrics: Dictionary of metric name to value.
            step: Training step number.
        """
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_system_metrics(self) -> None:
        """Log GPU utilization and memory metrics."""
        if not self.enabled:
            return
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1e9
                wandb.log({f"gpu_{i}_memory_gb": mem})

    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "model",
    ) -> None:
        """Log a file artifact (e.g. model checkpoint) to W&B.

        Args:
            path: Path to the file to upload.
            name: Artifact name.
            artifact_type: Type of artifact (e.g. 'model', 'dataset').
        """
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)
        logger.info("Logged artifact: %s (%s)", name, artifact_type)

    def create_scaling_chart(self, results: list[dict[str, Any]]) -> None:
        """Create a scaling efficiency chart in W&B.

        Args:
            results: List of dicts with 'num_gpus', 'speedup', 'efficiency'.
        """
        if not self.enabled:
            return
        table = wandb.Table(columns=["num_gpus", "speedup", "efficiency"])
        for r in results:
            table.add_data(r["num_gpus"], r["speedup"], r["efficiency"])
        wandb.log(
            {
                "scaling_chart": wandb.plot.line(
                    table, "num_gpus", "speedup", title="Scaling Speedup"
                )
            }
        )

    def finish(self) -> None:
        """End the W&B run."""
        if self.enabled:
            wandb.finish()
            logger.info("W&B run finished")
