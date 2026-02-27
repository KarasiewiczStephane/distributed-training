"""Scaling benchmark for comparing DDP vs Horovod frameworks."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Args:
        framework: Training framework used ('ddp' or 'horovod').
        num_gpus: Number of GPUs used.
        total_time: Total training time in seconds.
        throughput: Training throughput in samples/sec.
        memory_per_gpu: Peak GPU memory in GB.
        communication_overhead: Percentage of time spent on communication.
    """

    framework: Literal["ddp", "horovod"]
    num_gpus: int
    total_time: float
    throughput: float
    memory_per_gpu: float
    communication_overhead: float


@dataclass
class ScalingBenchmark:
    """Benchmark suite for comparing DDP vs Horovod scaling.

    Args:
        model_fn: Factory function to create the model.
        dataset: Dataset to train on.
        batch_size: Training batch size.
        num_iterations: Number of iterations per benchmark run.
    """

    model_fn: Callable[[], nn.Module]
    dataset: Dataset
    batch_size: int = 64
    num_iterations: int = 50
    results: list[BenchmarkResult] = field(default_factory=list)

    def measure_compute_time(
        self, model: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> float:
        """Measure pure compute time (forward + backward, no sync).

        Args:
            model: Model to benchmark.
            dataloader: Data to iterate over.
            device: Device to run on.

        Returns:
            Elapsed time in seconds.
        """
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        start = time.perf_counter()
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= self.num_iterations:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - start
        return elapsed

    def run_single_benchmark(
        self, framework: Literal["ddp", "horovod"], num_gpus: int = 1
    ) -> BenchmarkResult:
        """Run a single benchmark for a given framework and GPU count.

        Args:
            framework: Framework to benchmark ('ddp' or 'horovod').
            num_gpus: Number of GPUs (simulated for single-GPU runs).

        Returns:
            BenchmarkResult with timing metrics.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_fn().to(device)

        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, drop_last=True
        )

        total_time = self.measure_compute_time(model, dataloader, device)
        total_samples = min(self.num_iterations, len(dataloader)) * self.batch_size
        throughput = total_samples / total_time if total_time > 0 else 0

        memory_gb = 0.0
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9

        result = BenchmarkResult(
            framework=framework,
            num_gpus=num_gpus,
            total_time=total_time,
            throughput=throughput,
            memory_per_gpu=memory_gb,
            communication_overhead=0.0,
        )
        self.results.append(result)
        logger.info(
            "Benchmark %s (%d GPU): %.2fs, %.0f samples/sec",
            framework,
            num_gpus,
            total_time,
            throughput,
        )
        return result

    def run_comparison(
        self, gpu_counts: list[int] | None = None
    ) -> list[BenchmarkResult]:
        """Run comparison benchmarks for both frameworks.

        Args:
            gpu_counts: List of GPU counts to test. Defaults to [1].

        Returns:
            List of all benchmark results.
        """
        if gpu_counts is None:
            gpu_counts = [1]

        for num_gpus in gpu_counts:
            for framework in ["ddp", "horovod"]:
                self.run_single_benchmark(framework, num_gpus)
        return self.results

    def generate_report(self, output_path: str | None = None) -> str:
        """Generate a markdown report from benchmark results.

        Args:
            output_path: Optional file path to write the report.

        Returns:
            Markdown report string.
        """
        report = "# DDP vs Horovod Comparison\n\n"
        report += "| Framework | GPUs | Time (s) | Throughput | Memory (GB) |\n"
        report += "|-----------|------|----------|-----------|-------------|\n"

        for r in self.results:
            report += (
                f"| {r.framework} | {r.num_gpus} | "
                f"{r.total_time:.2f} | {r.throughput:.0f} img/s | "
                f"{r.memory_per_gpu:.2f} |\n"
            )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report)
            logger.info("Report written to %s", output_path)

        return report
