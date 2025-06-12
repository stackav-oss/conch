# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for benchmarking."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import triton


@dataclass
class BenchmarkMetadata:
    """Class holding metadata for benchmark."""

    platform: str
    params: dict[str, Any]


@dataclass
class BenchmarkResult:
    """Class holding results of benchmark."""

    tag: str
    metadata: BenchmarkMetadata
    num_iterations: int
    min_: float
    max_: float
    mean_: float
    median_: float

    def print_parameters(self, csv: bool = False) -> None:
        """Print the parameters of the benchmark.

        Args:
            csv: (Optional) If True, print in CSV format.
        """
        if csv:
            print(f"tag,platform,num_iterations,{','.join(self.metadata.params.keys())},runtime_ms")
            return

        print(f"Parameters: {self.metadata.params}")

    def csv_print(self) -> None:
        """Convert the benchmark result to a CSV string.

        Returns:
            A CSV string representation of the benchmark result.
        """
        print(
            f"{self.tag},{self.metadata.platform},{self.num_iterations},{','.join(str(v) for v in self.metadata.params.values())},{self.median_:.3f}"
        )

    def pretty_print(self) -> None:
        """Pretty print the benchmarking results.

        Args:
            unit: (Optional) Unit to print the times in, default is milliseconds.
        """

        def _format(value: float) -> str:
            return f"{value:.3f} ms"

        print(
            f"{self.tag}: num_iterations={self.num_iterations}, min={_format(self.min_)}, max={_format(self.max_)}, mean={_format(self.mean_)}, median={_format(self.median_)}"
        )

    def print_results(self, csv: bool = False) -> None:
        """Print the benchmark results.

        Args:
            csv: (Optional) If True, print in CSV format.
        """
        if csv:
            self.csv_print()
            return

        self.pretty_print()


def benchmark_it(
    fn: Callable[[], Any],
    tag: str,
    metadata: BenchmarkMetadata,
    iteration_time_ms: int = 10000,
    warmup_time_ms: int = 1000,
) -> BenchmarkResult:
    """Function to benchmark a function.

    Args:
        tag: The tag to identify the benchmark.
        fn: The function to benchmark.
        metadata: Metadata for the benchmark.
        iteration_time_ms: Time in milliseconds to run the benchmark.
        warmup_time_ms: Time in milliseconds to warm up before recording times.

    Returns:
        BenchmarkResult dataclass with aggregated results of benchmark.
    """
    results = triton.testing.do_bench(fn, warmup=warmup_time_ms, rep=iteration_time_ms, return_mode="all")
    num_iterations = len(results)

    return BenchmarkResult(
        tag,
        metadata,
        num_iterations,
        min(results),
        max(results),
        sum(results) / num_iterations,
        sorted(results)[num_iterations // 2],
    )
