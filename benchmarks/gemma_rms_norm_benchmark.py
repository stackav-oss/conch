# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch gemma_rms_norm benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_conch
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
@click.option(
    "--compile-ref",
    is_flag=True,
    help="Flag to torch.compile() the reference impl",
)
@click.option(
    "--compile-conch",
    is_flag=True,
    help="Flag to torch.compile() the Conch impl",
)
def main(
    embedding_size: int,
    num_tokens: int,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    compile_conch: bool,
) -> None:
    """Benchmark Conch GemmaRMSNorm op.

    Args:
        embedding_size: Embedding size.
        num_tokens: Number of tokens.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Conch.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
        compile_ref: Flag to torch.compile() the reference implementation.
        compile_conch: Flag to torch.compile() the Conch implementation.
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
    x_conch = x.clone()

    gemma_rms_norm_ref_fn = torch.compile(gemma_rms_norm_reference) if compile_ref else gemma_rms_norm_reference
    gemma_rms_norm_conch_fn = torch.compile(gemma_rms_norm_conch) if compile_conch else gemma_rms_norm_conch

    result_ref = gemma_rms_norm_ref_fn(x_ref, weights, epsilon, residual=None)
    result_conch = gemma_rms_norm_conch_fn(x_conch, weights, epsilon, residual=None)

    # For mypy (if residual==None then result is single Tensor, not tuple[Tensor, Tensor])
    assert isinstance(result_ref, torch.Tensor)
    assert isinstance(result_conch, torch.Tensor)

    if not torch.allclose(result_ref, result_conch, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(result_conch - result_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {result_ref}", file=sys.stderr)
            print(f"Conch output: {result_conch}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: gemma_rms_norm_ref_fn(x_ref, weights, epsilon, residual=None),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: gemma_rms_norm_conch_fn(x_conch, weights, epsilon, residual=None),
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
