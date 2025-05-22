# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton rms_norm benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.normalization.rms_norm import rms_norm as rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.rms_norm import rms_norm as rms_norm_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--hidden-size",
    required=True,
    type=int,
    default=2048,
    help="Dimension",
)
@click.option(
    "--num-tokens",
    required=True,
    type=int,
    default=4096,
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
    hidden_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton rms_norm op.

    Args:
        hidden_size: Dimension of input/output.
        num_tokens: Number of tokens.
        num_iterations: Number of iterations to record benchmark times for each impl
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16
    tolerance: Final = 1e-2
    epsilon: Final = 1e-6

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
        },
    )

    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    triton_output = rms_norm_triton(x, weight, epsilon)

    ref_output = rms_norm_reference(x, weight, epsilon)

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
        lambda: rms_norm_reference(
            x,
            weight,
            epsilon,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: rms_norm_triton(
            x,
            weight,
            epsilon,
        ),
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
