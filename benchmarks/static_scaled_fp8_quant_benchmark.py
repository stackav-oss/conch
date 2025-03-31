# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Static scaled fp8 quantization kernel benchmark."""

from typing import Final

import click
import torch

from conch.ops.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_triton
from conch.platforms import current_platform
from conch.reference.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import benchmark_it


def _dequantize(quantized_tensor: torch.Tensor, inv_scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a quantized tensor."""
    return quantized_tensor.to(dtype) * inv_scale


@click.command()
@click.option(
    "-h",
    "--hidden-size",
    required=True,
    type=int,
    default=4068,
    help="Hidden dimension",
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
    "-s",
    "--scale",
    required=True,
    type=float,
    default=2.1,
    help="Scaling arg",
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
    scale: float,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
) -> None:
    """Benchmark Triton static scaled FP8 quantization.

    Args:
        hidden_size: Hidden dimension of input tensors.
        num_tokens: Number of tokens in the input tensor.
        scale: Scaling factor to apply.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
    """
    if not current_platform.supports_fp8():
        error_msg = "FP8 not supported on this GPU, cannot run benchmark!"
        raise NotImplementedError(error_msg)

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    reference_output = scaled_fp8_quant_reference(x, scale_arg)
    triton_output, _ = scaled_fp8_quant_triton(x, scale_arg)

    if not torch.allclose(_dequantize(reference_output, scale, dtype), _dequantize(triton_output, scale, dtype)):
        print("WARNING: Reference and Triton results differ!")
        print(f"Output max diff: {(reference_output - triton_output).abs().max().item()}")

        if verbose:
            print(f"Reference output: {reference_output}")
            print(f"Triton output: {triton_output}")
    else:
        print("Results matched :)")

    # Benchmark Reference vs. Triton implementations
    baseline_result = benchmark_it(
        lambda: scaled_fp8_quant_reference(x, scale_arg),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: scaled_fp8_quant_triton(x, scale_arg),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    # Print results
    baseline_result.pretty_print(name="Baseline", unit="ms")
    triton_result.pretty_print(name="Triton", unit="ms")


if __name__ == "__main__":
    main()
