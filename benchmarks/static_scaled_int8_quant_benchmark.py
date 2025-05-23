# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Static scaled int8 quantization kernel benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.quantization.int8 import scaled_int8_quant as scaled_int8_quant_triton
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
def main(
    hidden_size: int,
    num_tokens: int,
    scale: float,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton static scaled int8 quantization.

    Args:
        hidden_size: Hidden dimension of input tensors.
        num_tokens: Number of tokens in the input tensor.
        scale: Scaling factor to apply.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
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
    triton_output, _ = scaled_int8_quant_triton(x, scale_arg)

    if not torch.allclose(ref_output, triton_output, atol=1, rtol=0.0):
        print("WARNING: Reference and Triton results differ!", file=sys.stderr)
        print(f"Output max diff: {(ref_output - triton_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Triton output: {triton_output}", file=sys.stderr)
    else:
        print("Results matched :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: scaled_int8_quant_reference(
            x,
            scale_arg,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: scaled_int8_quant_triton(
            x,
            scale_arg,
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
