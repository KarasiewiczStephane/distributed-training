"""Data loading benchmark to find optimal num_workers and measure throughput."""

import logging
import time

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def benchmark_num_workers(
    dataset: Dataset,
    batch_size: int,
    max_workers: int = 16,
    pin_memory: bool = True,
) -> list[dict[str, float]]:
    """Benchmark data loading speed across different num_workers values.

    Args:
        dataset: Dataset to benchmark loading from.
        batch_size: Batch size for the DataLoader.
        max_workers: Maximum number of workers to test.
        pin_memory: Whether to use pinned memory.

    Returns:
        List of dicts with num_workers, time, and throughput for each run.
    """
    worker_counts = [0, 2, 4, 8, 12, 16]
    results = []

    for num_workers in worker_counts:
        if num_workers > max_workers:
            break

        persistent = num_workers > 0
        prefetch = 2 if num_workers > 0 else None

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )

        start = time.perf_counter()
        for _ in loader:
            pass
        elapsed = time.perf_counter() - start

        throughput = len(dataset) / elapsed if elapsed > 0 else 0
        result = {
            "num_workers": num_workers,
            "time": elapsed,
            "throughput": throughput,
        }
        results.append(result)
        logger.info(
            "num_workers=%d: %.3fs (%.0f samples/sec)",
            num_workers,
            elapsed,
            throughput,
        )

    return results


def find_optimal_workers(
    dataset: Dataset,
    batch_size: int,
    max_workers: int = 16,
) -> int:
    """Find the optimal num_workers for fastest data loading.

    Args:
        dataset: Dataset to benchmark.
        batch_size: Batch size for the DataLoader.
        max_workers: Maximum workers to test.

    Returns:
        Optimal num_workers value.
    """
    results = benchmark_num_workers(dataset, batch_size, max_workers)
    best = min(results, key=lambda r: r["time"])
    logger.info(
        "Optimal num_workers=%d (%.3fs, %.0f samples/sec)",
        best["num_workers"],
        best["time"],
        best["throughput"],
    )
    return int(best["num_workers"])
