"""Automatic Mixed Precision (AMP) training mixin and benchmark utilities."""

import logging
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


class AMPMixin:
    """Mixin class that adds AMP support to any trainer.

    Provides autocast context management, gradient scaling, and
    a mixed-precision training step method.
    """

    def setup_amp(self, enabled: bool = True) -> None:
        """Initialize AMP components.

        Args:
            enabled: Whether to enable mixed precision.
                     Automatically disabled on CPU.
        """
        self.amp_enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.amp_enabled)
        logger.info("AMP %s", "enabled" if self.amp_enabled else "disabled")

    def get_autocast_context(self) -> Any:
        """Get the appropriate autocast context manager.

        Returns:
            torch.amp.autocast for GPU with AMP, nullcontext otherwise.
        """
        if self.amp_enabled:
            return autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train_step_amp(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_clip: float = 1.0,
    ) -> tuple[float, torch.Tensor]:
        """Execute a single training step with optional AMP.

        Args:
            inputs: Input batch tensor.
            targets: Target labels tensor.
            model: The neural network.
            criterion: Loss function.
            optimizer: Optimizer instance.
            gradient_clip: Maximum gradient norm.

        Returns:
            Tuple of (loss_value, model_outputs).
        """
        optimizer.zero_grad()
        with self.get_autocast_context():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        return loss.item(), outputs


def benchmark_amp(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_iterations: int = 50,
) -> dict[str, float]:
    """Benchmark FP32 vs AMP training throughput.

    Args:
        model: Neural network to benchmark.
        dataloader: DataLoader to iterate over.
        device: Device to run on.
        num_iterations: Number of iterations per run.

    Returns:
        Dictionary with fp32_time, amp_time, and speedup.
    """
    criterion = nn.CrossEntropyLoss()

    def run_iterations(use_amp: bool) -> float:
        m = model.to(device)
        m.train()
        optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
        scaler = GradScaler(enabled=use_amp)
        if use_amp:
            ctx = autocast(device_type="cuda", dtype=torch.float16)
        else:
            ctx = nullcontext()

        start = time.perf_counter()
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_iterations:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with ctx:
                outputs = m(inputs)
                loss = criterion(outputs, targets)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        elapsed = time.perf_counter() - start
        return elapsed

    fp32_time = run_iterations(use_amp=False)
    amp_time = run_iterations(use_amp=True)
    speedup = fp32_time / amp_time if amp_time > 0 else 0

    logger.info(
        "AMP Benchmark: FP32=%.2fs, AMP=%.2fs, Speedup=%.2fx",
        fp32_time,
        amp_time,
        speedup,
    )
    return {"fp32_time": fp32_time, "amp_time": amp_time, "speedup": speedup}
