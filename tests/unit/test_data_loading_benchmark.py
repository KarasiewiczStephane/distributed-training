"""Tests for data loading benchmark utilities."""

import torch
from torch.utils.data import TensorDataset

from src.benchmarks.data_loading import benchmark_num_workers, find_optimal_workers


def _make_dataset(n: int = 128):
    """Create a small synthetic dataset."""
    return TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,)))


class TestBenchmarkNumWorkers:
    """Tests for benchmark_num_workers function."""

    def test_returns_list_of_results(self):
        """Returns a list of dicts with expected keys."""
        dataset = _make_dataset()
        results = benchmark_num_workers(dataset, batch_size=16, max_workers=0)
        assert len(results) == 1
        assert "num_workers" in results[0]
        assert "time" in results[0]
        assert "throughput" in results[0]

    def test_respects_max_workers(self):
        """Only tests up to max_workers."""
        dataset = _make_dataset()
        results = benchmark_num_workers(dataset, batch_size=16, max_workers=2)
        workers_tested = [r["num_workers"] for r in results]
        assert all(w <= 2 for w in workers_tested)

    def test_time_is_positive(self):
        """All timing results are positive."""
        dataset = _make_dataset()
        results = benchmark_num_workers(dataset, batch_size=16, max_workers=0)
        assert results[0]["time"] > 0


class TestFindOptimalWorkers:
    """Tests for find_optimal_workers function."""

    def test_returns_int(self):
        """Returns an integer num_workers value."""
        dataset = _make_dataset()
        optimal = find_optimal_workers(dataset, batch_size=16, max_workers=0)
        assert isinstance(optimal, int)
        assert optimal >= 0
