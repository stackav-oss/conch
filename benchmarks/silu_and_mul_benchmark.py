# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch silu_and_mul benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.activation.silu_and_mul import silu_and_mul as silu_and_mul_conch
from conch.platforms import current_platform
from conch.reference.activation.silu_and_mul import silu_and_mul as silu_and_mul_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--dim",
    required=True,
    type=int,
    default=256,
    help="Dimension",
)
@click.option(
    "--batch-size",
    required=True,
    type=int,
    default=32,
    help="Number of tokens",
)
@click.option(
    "--num-tokens",
    required=True,
    type=int,
    default=2048,
    help="Number of tokens",
)
@click.option(
    "--iteration-time-ms",
    required=False,
    type=int,
    default=10000,
    help="Time in milliseconds to run benchmark",
)
@click.option(
    "--warmup-time-ms",
    required=False,
    type=int,
    default=1000,
    help="Time in milliseconds to warmup before recording times",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Flag for printing verbose output",
)
@click.option(
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
@click.option(
    "--csv",
    is_flag=True,
    help="Flag for printing results in CSV format",
)
def main(  # noqa: PLR0913
    dim: int,
    batch_size: int,
    num_tokens: int,
    iteration_time_ms: int,
    warmup_time_ms: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark silu_and_mul op.

    Args:
        dim: Dimension of input/output.
        batch_size: Batch size.
        num_tokens: Number of tokens.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16
    tolerance: Final = 1e-3

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "dim": dim,
            "batch_size": batch_size,
            "num_tokens": num_tokens,
        },
    )

    x_shape = (batch_size, num_tokens, 2 * dim)
    x = torch.randn(x_shape, dtype=dtype, device=device)

    ref_output = silu_and_mul_reference(x)
    conch_output = silu_and_mul_conch(x)

    if not torch.allclose(ref_output, conch_output, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and conch results differ! (atol={tolerance}, rtol={tolerance})", file=sys.stderr)
        print(f"Output max diff: {(ref_output - conch_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print(f"Results matched with atol={tolerance} and rtol={tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. conch implementations
    baseline_result = benchmark_it(
        lambda: silu_and_mul_reference(x),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: silu_and_mul_conch(x),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    # Print results
    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
