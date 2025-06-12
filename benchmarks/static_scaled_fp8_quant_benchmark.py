# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Static scaled fp8 quantization kernel benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_conch
from conch.platforms import current_platform
from conch.reference.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _dequantize(quantized_tensor: torch.Tensor, inv_scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a quantized tensor."""
    return quantized_tensor.to(dtype) * inv_scale


@click.command()
@click.option(
    "--hidden-size",
    required=True,
    type=int,
    default=4068,
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
    """Benchmark static scaled FP8 quantization.

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
    if not current_platform.supports_fp8():
        error_msg = "FP8 not supported on this GPU, cannot run benchmark!"
        raise NotImplementedError(error_msg)

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

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    reference_output = scaled_fp8_quant_reference(x, scale_arg)
    conch_output, _ = scaled_fp8_quant_conch(x, scale_arg)

    if not torch.allclose(_dequantize(reference_output, scale, dtype), _dequantize(conch_output, scale, dtype)):
        print("WARNING: Reference and Conch results differ!", file=sys.stderr)
        print(f"Output max diff: {(reference_output - conch_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {reference_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print("Results matched :)", file=sys.stderr)

    # Benchmark Reference vs. Conch implementations
    baseline_result = benchmark_it(
        lambda: scaled_fp8_quant_reference(x, scale_arg),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: scaled_fp8_quant_conch(x, scale_arg),
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
