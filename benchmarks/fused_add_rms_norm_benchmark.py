# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton rms_norm benchmark."""

from typing import Final

import click
import torch

from conch.ops.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import benchmark_it


@click.command()
@click.option(
    "-d",
    "--hidden-size",
    required=True,
    type=int,
    default=2048,
    help="Dimension",
)
@click.option(
    "-t",
    "--num-tokens",
    required=True,
    type=int,
    default=4096,
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
def main(  # noqa: PLR0913
    hidden_size: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
) -> None:
    """Benchmark Triton rms_norm op.

    Args:
        hidden_size: Dimension of input/output.
        num_tokens: Number of tokens.
        num_iterations: Number of iterations to record benchmark times for each impl
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16
    tolerance: Final = 1e-2
    epsilon: Final = 1e-6

    x_shape = (num_tokens, hidden_size)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.randn(x_shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    triton_x = x.clone()
    ref_x = x.clone()
    triton_residual = residual.clone()
    ref_residual = residual.clone()

    triton_output, triton_residual = fused_add_rms_norm_triton(triton_x, triton_residual, weight, epsilon)

    ref_output, ref_residual = fused_add_rms_norm_reference(ref_x, ref_residual, weight, epsilon)

    if not torch.allclose(ref_output, triton_output, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={tolerance}, rtol={tolerance})")
        print(f"Output max diff: {(ref_output - triton_output).abs().max().item()}")

        if verbose:
            print(f"Reference output: {ref_output}")
            print(f"Triton output: {triton_output}")
    else:
        print(f"Results matched with atol={tolerance} and rtol={tolerance} :)")

    if not torch.allclose(ref_residual, triton_residual, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and Triton residuals differ! (atol={tolerance}, rtol={tolerance})")
        print(f"Output max diff: {(ref_residual - triton_residual).abs().max().item()}")

        if verbose:
            print(f"Reference output: {ref_residual}")
            print(f"Triton output: {triton_residual}")
    else:
        print(f"Residuals matched with atol={tolerance} and rtol={tolerance} :)")

    # Benchmark Reference vs. Triton implementations
    baseline_result = benchmark_it(
        lambda: fused_add_rms_norm_reference(
            ref_x,
            ref_residual,
            weight,
            epsilon,
        ),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: fused_add_rms_norm_triton(
            triton_x,
            triton_residual,
            weight,
            epsilon,
        ),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    # Print results
    baseline_result.pretty_print(name="Baseline", unit="ms")
    triton_result.pretty_print(name="Triton", unit="ms")


if __name__ == "__main__":
    main()
