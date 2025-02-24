# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Utilities for benchmarking."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import torch


@dataclass
class BenchmarkResult:
    """Class holding results of benchmark."""

    num_iterations: int
    min_: float
    max_: float
    mean_: float
    median_: float

    def pretty_print(self, name: str | None = None, unit: str = "s") -> None:
        """Pretty print the benchmarking results.

        Args:
            name: (Optional) Name to identify these results.
            unit: (Optional) Unit to print the times in, default is seconds.
        """
        # Baseline, no conversion
        conversion_factor: float | None = None

        if unit == "ms":
            conversion_factor = 1.0e3
        if unit == "us":
            conversion_factor = 1.0e6
        if unit == "ns":
            conversion_factor = 1.0e9

        def _format(value: float) -> str:
            if conversion_factor is None:
                return f"{value} {unit}"
            return f"{value * conversion_factor:.3f} {unit}"

        prefix = f"{name}: " if name is not None else ""
        print(
            f"{prefix}num_iterations={self.num_iterations}, min={_format(self.min_)}, max={_format(self.max_)}, mean={_format(self.mean_)}, median={_format(self.median_)}"
        )


def _benchmark_wall(fn: Callable[[], Any]) -> float:
    """Record the time of a function with wall clock.

    Args:
        fn: The function to execute.

    Returns:
        The wall clock execution time.
    """
    start_time = time.time()

    fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    return end_time - start_time


def _benchmark_cuda_event(fn: Callable[[], Any]) -> float:
    """Record the time of a function with CUDA events.

    Args:
        fn: The function to execute.

    Returns:
        The elapsed time between CUDA events at the start/end of the function.
    """
    _ms_to_s: Final = 1.0e-3

    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

    start.record()  # type: ignore[no-untyped-call]

    fn()

    end.record()  # type: ignore[no-untyped-call]

    torch.cuda.synchronize()
    return start.elapsed_time(end) * _ms_to_s  # type: ignore[no-untyped-call, no-any-return]


def benchmark_it(
    fn: Callable[[], Any],
    num_iterations: int = 100,
    num_warmup_iterations: int = 10,
    device: torch.device | str = "cuda",
    benchmark_type: str = "cuda_event",
) -> BenchmarkResult:
    """Function to benchmark wall clock time for a function.

    Args:
        fn: The function to benchmark.
        num_iterations: (Optional) Number of times to run fn.
        num_warmup_iterations: (Optional) Number of times to run fn to "warmup".
        device: (Optional) Torch device to allocate tensor for clearing L2 cache.
        benchmark_type: (Optional) Timing backend to use for calculating elapsed time, either wall clock or CUDA event.

    Raises:
        NotImplementedError: If benchmark_type not in the supported list.

    Returns:
        BenchmarkResult dataclass with aggregated results of benchmark.
    """
    _supported_benchmark_types: Final = ["wall", "cuda_event"]

    if benchmark_type not in _supported_benchmark_types:
        msg = f"benchmark_type '{benchmark_type}' not in list of supported types: {_supported_benchmark_types}"
        raise NotImplementedError(msg)

    # We maintain a buffer of 256 MB that we clear before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size: Final = int(256 * 1024 * 1024)
    cache = torch.empty(cache_size, dtype=torch.int8, device=device)

    benchmark_fn: Callable[[Callable[[], Any]], float] = _benchmark_cuda_event
    if benchmark_type == "wall":
        benchmark_fn = _benchmark_wall

    # Warm-up
    for _ in range(num_warmup_iterations):
        _ = benchmark_fn(fn)

    results = []

    # Benchmark
    for _ in range(num_iterations):
        # Clear the L2 cache before each run
        cache.zero_()
        results.append(benchmark_fn(fn))

    return BenchmarkResult(
        num_iterations,
        min(results),
        max(results),
        sum(results) / num_iterations,
        sorted(results)[num_iterations // 2],
    )
