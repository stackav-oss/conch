# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton gemma_rms_norm benchmark."""

from typing import Final

import click
import torch

from conch.ops.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import benchmark_it


@click.command()
@click.option(
    "-s",
    "--embedding-size",
    required=False,
    type=int,
    default=2048,
    help="Embedding size",
)
@click.option(
    "-t",
    "--num-tokens",
    required=False,
    type=int,
    default=8192,
    help="Number of tokens",
)
@click.option(
    "-i",
    "--num-iterations",
    required=False,
    type=int,
    default=100,
    help="Number of iterations",
)
@click.option(
    "-w",
    "--num-warmup-iterations",
    required=False,
    type=int,
    default=10,
    help="Number of warmup iterations",
)
@click.option(
    "-a",
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-2,
    help="Absolute tolerance to match with",
)
@click.option(
    "-v",
    "--verbose",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Flag for printing verbose output",
)
@click.option(
    "-g",
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
def main(
    embedding_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
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
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    epsilon: Final = 1e-6

    x = torch.randn((num_tokens, embedding_size), dtype=torch.float16, device=device)
    weights = torch.randn((embedding_size,), device=device)

    x_ref = x.clone()
    x_triton = x.clone()

    result_ref = gemma_rms_norm_reference(x_ref, weights, epsilon, residual=None)
    result_triton = gemma_rms_norm_triton(x_triton, weights, epsilon, residual=None)

    if not torch.allclose(result_ref, result_triton, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})")
        print(f"Output max diff: {(result_triton - result_ref).abs().max().item()}")

        if verbose:
            print(f"Reference output: {result_ref}")
            print(f"Triton output: {result_triton}")
    else:
        print(f"Results matched with atol={absolute_tolerance} :)")

    baseline_result = benchmark_it(
        lambda: gemma_rms_norm_reference(x_ref, weights, epsilon, residual=None),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: gemma_rms_norm_triton(x_triton, weights, epsilon, residual=None),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    baseline_result.pretty_print(name="Baseline", unit="ms")
    triton_result.pretty_print(name="Triton", unit="ms")


if __name__ == "__main__":
    main()
