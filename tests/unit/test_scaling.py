"""Tests for the scaling benchmark module."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.benchmarks.scaling import BenchmarkResult, ScalingBenchmark


def _model_fn():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))


def _make_dataset(n: int = 128):
    return TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_create_result(self):
        """BenchmarkResult stores all fields correctly."""
        r = BenchmarkResult(
            framework="ddp",
            num_gpus=2,
            total_time=10.0,
            throughput=1000.0,
            memory_per_gpu=2.5,
            communication_overhead=5.0,
        )
        assert r.framework == "ddp"
        assert r.num_gpus == 2
        assert r.throughput == 1000.0


class TestScalingBenchmark:
    """Tests for the ScalingBenchmark class."""

    def test_single_benchmark(self):
        """run_single_benchmark produces a valid result."""
        bench = ScalingBenchmark(
            model_fn=_model_fn,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        result = bench.run_single_benchmark("ddp", num_gpus=1)
        assert result.framework == "ddp"
        assert result.total_time > 0
        assert result.throughput > 0

    def test_run_comparison(self):
        """run_comparison produces results for each framework/gpu combo."""
        bench = ScalingBenchmark(
            model_fn=_model_fn,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        results = bench.run_comparison(gpu_counts=[1])
        assert len(results) == 2  # ddp + horovod
        frameworks = {r.framework for r in results}
        assert "ddp" in frameworks
        assert "horovod" in frameworks

    def test_generate_report(self):
        """generate_report produces valid markdown."""
        bench = ScalingBenchmark(
            model_fn=_model_fn,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        bench.run_comparison(gpu_counts=[1])
        report = bench.generate_report()
        assert "DDP vs Horovod" in report
        assert "Framework" in report
        assert "ddp" in report

    def test_generate_report_to_file(self, tmp_path):
        """generate_report writes to file when path is given."""
        bench = ScalingBenchmark(
            model_fn=_model_fn,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        bench.run_single_benchmark("ddp", num_gpus=1)
        out_path = str(tmp_path / "report.md")
        report = bench.generate_report(output_path=out_path)
        assert (tmp_path / "report.md").exists()
        assert "ddp" in report
