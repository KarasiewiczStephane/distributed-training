"""Tests for ScalingBenchmarkSuite and run_benchmark CLI."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.benchmarks.scaling import ScalingBenchmarkSuite


def _make_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))


def _make_dataset(n=64):
    return TensorDataset(torch.randn(n, 3, 4, 4), torch.randint(0, 10, (n,)))


class TestScalingBenchmarkSuite:
    """Tests for the ScalingBenchmarkSuite class."""

    def test_speedup_calculation(self):
        """Speedup for 1 GPU should be 1.0x."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        results = suite.run_scaling_benchmark(gpu_counts=[1])
        assert len(results) == 1
        assert results[0]["speedup"] == 1.0

    def test_efficiency_formula(self):
        """Efficiency = speedup / num_gpus."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        results = suite.run_scaling_benchmark(gpu_counts=[1, 2])
        for r in results:
            expected = r["speedup"] / r["num_gpus"]
            assert abs(r["efficiency"] - expected) < 1e-6

    def test_multiple_gpu_counts(self):
        """Results are generated for each GPU count."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        results = suite.run_scaling_benchmark(gpu_counts=[1, 2, 4])
        assert len(results) == 3
        gpus = [r["num_gpus"] for r in results]
        assert gpus == [1, 2, 4]

    def test_results_contain_required_keys(self):
        """Each result has num_gpus, time, speedup, efficiency keys."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        results = suite.run_scaling_benchmark(gpu_counts=[1])
        for r in results:
            assert "num_gpus" in r
            assert "time" in r
            assert "speedup" in r
            assert "efficiency" in r


class TestMarkdownReport:
    """Tests for markdown report generation."""

    def test_report_contains_header(self):
        """Report starts with proper markdown header."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        suite.run_scaling_benchmark(gpu_counts=[1])
        report = suite.generate_markdown_report()
        assert "# Distributed Training Benchmark Results" in report

    def test_report_contains_table(self):
        """Report contains a markdown table with results."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        suite.run_scaling_benchmark(gpu_counts=[1])
        report = suite.generate_markdown_report()
        assert "| GPUs | Time (s) | Speedup | Efficiency |" in report
        assert "| 1 |" in report

    def test_report_written_to_file(self, tmp_path):
        """Report is written to the output file."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        suite.run_scaling_benchmark(gpu_counts=[1])
        out = str(tmp_path / "report.md")
        report = suite.generate_markdown_report(output_path=out)
        assert (tmp_path / "report.md").read_text() == report

    def test_report_creates_parent_dirs(self, tmp_path):
        """Report creates parent directories if needed."""
        suite = ScalingBenchmarkSuite(
            model_fn=_make_model,
            dataset=_make_dataset(),
            batch_size=16,
            num_iterations=2,
        )
        suite.run_scaling_benchmark(gpu_counts=[1])
        out = str(tmp_path / "sub" / "dir" / "report.md")
        suite.generate_markdown_report(output_path=out)
        assert (tmp_path / "sub" / "dir" / "report.md").exists()


class TestRunBenchmarkCLI:
    """Tests for run_benchmark module."""

    def test_parse_args_defaults(self):
        """Default args are set correctly."""
        from unittest.mock import patch

        from src.benchmarks.run_benchmark import parse_args

        with patch("sys.argv", ["run_benchmark.py"]):
            args = parse_args()
        assert args.gpus == [1, 2, 4]
        assert args.output == "reports/benchmark.md"
        assert args.mode == "all"

    def test_create_dummy_model(self):
        """Dummy model produces correct output shape."""
        from src.benchmarks.run_benchmark import create_dummy_model

        model = create_dummy_model()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 100)

    def test_create_dummy_dataset(self):
        """Dummy dataset has expected size."""
        from src.benchmarks.run_benchmark import create_dummy_dataset

        ds = create_dummy_dataset()
        assert len(ds) == 512
