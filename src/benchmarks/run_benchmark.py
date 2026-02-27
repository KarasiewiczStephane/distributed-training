"""CLI entry point for running benchmarks and generating reports."""

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.benchmarks.scaling import ScalingBenchmark, ScalingBenchmarkSuite

logger = logging.getLogger(__name__)


def create_dummy_model() -> nn.Module:
    """Create a simple model for benchmarking."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )


def create_dummy_dataset() -> TensorDataset:
    """Create a synthetic dataset for benchmarking."""
    inputs = torch.randn(512, 3, 32, 32)
    targets = torch.randint(0, 100, (512,))
    return TensorDataset(inputs, targets)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run distributed training benchmarks")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="GPU counts to benchmark",
    )
    parser.add_argument(
        "--output",
        default="reports/benchmark.md",
        help="Output path for markdown report",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of training iterations per benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--mode",
        choices=["scaling", "comparison", "all"],
        default="all",
        help="Benchmark mode to run",
    )
    return parser.parse_args()


def main() -> None:
    """Run benchmarks based on CLI arguments."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = create_dummy_dataset()

    if args.mode in ("scaling", "all"):
        suite = ScalingBenchmarkSuite(
            model_fn=create_dummy_model,
            dataset=dataset,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
        )
        suite.run_scaling_benchmark(gpu_counts=args.gpus)
        report = suite.generate_markdown_report(output_path=args.output)
        print(report)

    if args.mode in ("comparison", "all"):
        bench = ScalingBenchmark(
            model_fn=create_dummy_model,
            dataset=dataset,
            batch_size=args.batch_size,
            num_iterations=args.iterations,
        )
        bench.run_comparison(gpu_counts=args.gpus)
        comparison_path = args.output.replace(".md", "_comparison.md")
        report = bench.generate_report(output_path=comparison_path)
        print(report)


if __name__ == "__main__":
    main()
