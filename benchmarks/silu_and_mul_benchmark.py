# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton silu_and_mul benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.activation.silu_and_mul import silu_and_mul as silu_and_mul_triton
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
    "--num-iterations",
    required=False,
    type=int,
    default=100,
    help="Number of iterations",
)
@click.option(
    "--num-warmup-iterations",
    required=False,
    type=int,
    default=10,
    help="Number of warmup iterations",
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
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton silu_and_mul op.

    Args:
        dim: Dimension of input/output.
        batch_size: Batch size.
        num_tokens: Number of tokens.
        num_iterations: Number of iterations to record benchmark times for each impl
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times
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
    triton_output = silu_and_mul_triton(x)

    if not torch.allclose(ref_output, triton_output, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={tolerance}, rtol={tolerance})", file=sys.stderr)
        print(f"Output max diff: {(ref_output - triton_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Triton output: {triton_output}", file=sys.stderr)
    else:
        print(f"Results matched with atol={tolerance} and rtol={tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. Triton implementations
    baseline_result = benchmark_it(
        lambda: silu_and_mul_reference(x),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: silu_and_mul_triton(x),
        tag="Triton",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    # Print results
    triton_result.print_parameters(csv=csv)
    triton_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
