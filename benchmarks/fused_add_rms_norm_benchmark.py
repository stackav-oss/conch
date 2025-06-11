# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch rms_norm benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_conch
from conch.platforms import current_platform
from conch.reference.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_reference
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
    hidden_size: int,
    num_tokens: int,
    iteration_time_ms: int,
    warmup_time_ms: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Conch rms_norm op.

    Args:
        hidden_size: Dimension of input/output.
        num_tokens: Number of tokens.
        iteration_time_ms: Time in milliseconds to run the benchmark.
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
    tolerance: Final = 1e-2
    epsilon: Final = 1e-6

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
        },
    )

    x_shape = (num_tokens, hidden_size)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.randn(x_shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    conch_x = x.clone()
    ref_x = x.clone()
    conch_residual = residual.clone()
    ref_residual = residual.clone()

    conch_output, conch_residual = fused_add_rms_norm_conch(conch_x, conch_residual, weight, epsilon)

    ref_output, ref_residual = fused_add_rms_norm_reference(ref_x, ref_residual, weight, epsilon)

    if not torch.allclose(ref_output, conch_output, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={tolerance}, rtol={tolerance})", file=sys.stderr)
        print(f"Output max diff: {(ref_output - conch_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print(f"Results matched with atol={tolerance} and rtol={tolerance} :)", file=sys.stderr)

    if not torch.allclose(ref_residual, conch_residual, atol=tolerance, rtol=tolerance):
        print(f"WARNING: Reference and Conch residuals differ! (atol={tolerance}, rtol={tolerance})", file=sys.stderr)
        print(f"Output max diff: {(ref_residual - conch_residual).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_residual}", file=sys.stderr)
            print(f"Conch output: {conch_residual}", file=sys.stderr)
    else:
        print(f"Residuals matched with atol={tolerance} and rtol={tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. Conch implementations
    baseline_result = benchmark_it(
        lambda: fused_add_rms_norm_reference(
            ref_x,
            ref_residual,
            weight,
            epsilon,
        ),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: fused_add_rms_norm_conch(
            conch_x,
            conch_residual,
            weight,
            epsilon,
        ),
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
