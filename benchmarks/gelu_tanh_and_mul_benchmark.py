# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton gelu_tanh_and_mul benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_triton
from conch.platforms import current_platform
from conch.reference.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--hidden-size",
    required=False,
    type=int,
    default=13824,
    help="Feedforward hidden size",
)
@click.option(
    "--num-tokens",
    required=False,
    type=int,
    default=8192,
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
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-3,
    help="Absolute tolerance to match with",
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
def main(
    hidden_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton GeluTanhAndMul op.

    Args:
        hidden_size: Feedforward hidden size.
        num_tokens: Number of tokens.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
        },
    )

    projections = torch.rand((num_tokens, hidden_size * 2), device=device)

    ref_output = gelu_tanh_and_mul_reference(projections)
    triton_output = gelu_tanh_and_mul_triton(projections)

    if not torch.allclose(ref_output, triton_output, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(triton_output - ref_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Triton output: {triton_output}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: gelu_tanh_and_mul_reference(projections),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: gelu_tanh_and_mul_triton(projections),
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
