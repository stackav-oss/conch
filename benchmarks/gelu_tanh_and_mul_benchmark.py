# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton gelu_tanh_and_mul benchmark."""

from typing import Final

import click
import torch

from conch.ops.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_triton
from conch.platforms import current_platform
from conch.reference.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import benchmark_it


@click.command()
@click.option(
    "-s",
    "--hidden-size",
    required=False,
    type=int,
    default=13824,
    help="Feedforward hidden size",
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
    default=1e-3,
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
    hidden_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
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
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    projections = torch.rand((num_tokens, hidden_size * 2), device=device)

    ref_output = gelu_tanh_and_mul_reference(projections)
    triton_output = gelu_tanh_and_mul_triton(projections)

    if not torch.allclose(ref_output, triton_output, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})")
        print(f"Output max diff: {(triton_output - ref_output).abs().max().item()}")

        if verbose:
            print(f"Reference output: {ref_output}")
            print(f"Triton output: {triton_output}")
    else:
        print(f"Results matched with atol={absolute_tolerance} :)")

    baseline_result = benchmark_it(
        lambda: gelu_tanh_and_mul_reference(projections),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: gelu_tanh_and_mul_triton(projections),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    baseline_result.pretty_print(name="Baseline", unit="ms")
    triton_result.pretty_print(name="Triton", unit="ms")


if __name__ == "__main__":
    main()
