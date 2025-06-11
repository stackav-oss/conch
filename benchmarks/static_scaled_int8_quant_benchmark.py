# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Static scaled int8 quantization kernel benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.quantization.int8 import scaled_int8_quant as scaled_int8_quant_conch
from conch.platforms import current_platform
from conch.reference.quantization.int8 import scaled_int8_quant as scaled_int8_quant_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--hidden-size",
    required=True,
    type=int,
    default=4608,
    help="Hidden dimension",
)
@click.option(
    "--num-tokens",
    required=True,
    type=int,
    default=4096,
    help="Number of tokens",
)
@click.option(
    "--scale",
    required=True,
    type=float,
    default=2.1,
    help="Scaling arg",
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
def main(
    hidden_size: int,
    num_tokens: int,
    scale: float,
    iteration_time_ms: int,
    warmup_time_ms: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark static scaled int8 quantization.

    Args:
        hidden_size: Hidden dimension of input tensors.
        num_tokens: Number of tokens in the input tensor.
        scale: Scaling factor to apply.
        iteration_time_ms: Time in milliseconds to run the benchmark.
        warmup_time_ms: Time in milliseconds to warm up before recording times.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "hidden_size": hidden_size,
            "num_tokens": num_tokens,
        },
    )

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    ref_output = scaled_int8_quant_reference(x, scale_arg)
    conch_output, _ = scaled_int8_quant_conch(x, scale_arg)

    if not torch.allclose(ref_output, conch_output, atol=1, rtol=0.0):
        print("WARNING: Reference and Conch results differ!", file=sys.stderr)
        print(f"Output max diff: {(ref_output - conch_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print("Results matched :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: scaled_int8_quant_reference(
            x,
            scale_arg,
        ),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: scaled_int8_quant_conch(
            x,
            scale_arg,
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
