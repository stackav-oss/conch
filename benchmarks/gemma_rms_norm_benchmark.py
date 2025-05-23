# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton gemma_rms_norm benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--embedding-size",
    required=False,
    type=int,
    default=2048,
    help="Embedding size",
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
    default=1e-2,
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
    embedding_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton GemmaRMSNorm op.

    Args:
        embedding_size: Embedding size.
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

    epsilon: Final = 1e-6

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "embedding_size": embedding_size,
            "num_tokens": num_tokens,
        },
    )

    x = torch.randn((num_tokens, embedding_size), dtype=torch.float16, device=device)
    weights = torch.randn((embedding_size,), device=device)

    x_ref = x.clone()
    x_triton = x.clone()

    result_ref = gemma_rms_norm_reference(x_ref, weights, epsilon, residual=None)
    result_triton = gemma_rms_norm_triton(x_triton, weights, epsilon, residual=None)

    # For mypy (if residual==None then result is single Tensor, not tuple[Tensor, Tensor])
    assert isinstance(result_ref, torch.Tensor)
    assert isinstance(result_triton, torch.Tensor)

    if not torch.allclose(result_ref, result_triton, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(result_triton - result_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {result_ref}", file=sys.stderr)
            print(f"Triton output: {result_triton}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: gemma_rms_norm_reference(x_ref, weights, epsilon, residual=None),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: gemma_rms_norm_triton(x_triton, weights, epsilon, residual=None),
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
