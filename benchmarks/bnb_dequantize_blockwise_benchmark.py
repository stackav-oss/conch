# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Bitsandbytes dequantize blockwise benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.quantization.bitsandbytes.functional import dequantize_4bit as triton_dequantize_4bit
from conch.ops.quantization.bitsandbytes.functional import quantize_4bit as triton_quantize_4bit
from conch.platforms import current_platform
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map click arg for dtype to torch type."""
    if dtype_str == "uint8":
        return torch.uint8
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32

    error_msg = f"Unrecognized data type: '{dtype_str}'"
    raise ValueError(error_msg)


@click.command()
@click.option(
    "--blocksize",
    required=True,
    type=int,
    default=64,
    help="Size of quantized blocks",
)
@click.option(
    "--size-multiplier",
    required=True,
    type=float,
    default=458752,
    help="How many quantized blocks are in the input tensor",
)
@click.option(
    "--quant-type",
    required=True,
    type=click.Choice(["nf4", "fp4"]),
    default="nf4",
    help="Data type of quantized input",
)
@click.option(
    "--dequant-dtype",
    required=True,
    type=click.Choice(["fp16", "bf16", "fp32"]),
    default="bf16",
    help="Data type before quantization",
)
@click.option(
    "--quant-storage-dtype",
    required=True,
    type=click.Choice(["uint8", "fp16", "bf16", "fp32"]),
    default="uint8",
    help="Data type for storing quantized results",
)
@click.option(
    "--compress-statistics",
    is_flag=True,
    help="Flag for double-quantization",
)
@click.option(
    "--enable-bnb",
    is_flag=True,
    default=envs.CONCH_BENCH_ENABLE_ALL_REF,
    help="Flag to enable BNB reference impl",
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
def main(  # noqa: PLR0913
    blocksize: int,
    size_multiplier: float,
    quant_type: str,
    dequant_dtype: str,
    quant_storage_dtype: str,
    compress_statistics: bool,
    enable_bnb: bool,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton blockwise quantization kernel.

    Args:
        blocksize: Size of quantized blocks.
        size_multiplier: How many quantized blocks are in the input tensor.
        quant_type: Data type of quantized input.
        dequant_dtype: Data type before quantization.
        quant_storage_dtype: Data type for storing quantized results.
        compress_statistics: Flag for double-quantization.
        enable_bnb: Flag to enable bitsandbytes reference implementation.
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

    dtype: Final = _to_torch_dtype(dequant_dtype)
    quant_storage: Final = _to_torch_dtype(quant_storage_dtype)

    input_size = int(blocksize * size_multiplier)
    x = torch.randn((input_size,), device=device, dtype=dtype)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "blocksize": blocksize,
            "size_multiplier": size_multiplier,
            "quant_type": quant_type,
            "dequant_dtype": dequant_dtype,
            "quant_storage_dtype": quant_storage_dtype,
            "compress_statistics": compress_statistics,
        },
    )

    triton_quantized, triton_state = triton_quantize_4bit(
        x,
        absmax=None,
        out=None,
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
        quant_storage=quant_storage,
    )

    triton_dequantized = triton_dequantize_4bit(
        triton_quantized,
        quant_state=triton_state,
        absmax=None,
        out=None,
        blocksize=blocksize,
        quant_type=quant_type,
    )

    if enable_bnb:
        if not envs.CONCH_ENABLE_BNB:
            error_msg = "bitsandbytes must be installed and enabled via CONCH_ENABLE_BNB=1"
            raise NotImplementedError(error_msg)

        from bitsandbytes.functional import dequantize_4bit as bnb_dequantize_4bit
        from bitsandbytes.functional import quantize_4bit as bnb_quantize_4bit

        bnb_quantized, bnb_state = bnb_quantize_4bit(
            x,
            absmax=None,
            out=None,
            blocksize=blocksize,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
        )

        if not torch.allclose(triton_quantized, bnb_quantized):
            print("WARNING: Bitsandbytes and Triton results differ!", file=sys.stderr)
            print(f"Output max diff: {(bnb_quantized - triton_quantized).abs().max().item()}", file=sys.stderr)

            if verbose:
                print(f"Torch output: {bnb_quantized}", file=sys.stderr)
                print(f"Triton output: {triton_quantized}", file=sys.stderr)

        bnb_dequantized = bnb_dequantize_4bit(
            bnb_quantized,
            quant_state=bnb_state,
            absmax=None,
            out=None,
            blocksize=blocksize,
            quant_type=quant_type,
        )

        if not torch.allclose(triton_dequantized, bnb_dequantized):
            print("WARNING: Bitsandbytes and Triton results differ!", file=sys.stderr)
            print(f"Output max diff: {(bnb_dequantized - triton_dequantized).abs().max().item()}", file=sys.stderr)

            if verbose:
                print(f"Torch output: {bnb_dequantized}", file=sys.stderr)
                print(f"Triton output: {triton_dequantized}", file=sys.stderr)
        else:
            print("Results matched :)", file=sys.stderr)

        baseline_result = benchmark_it(
            lambda: bnb_dequantize_4bit(
                bnb_quantized,
                quant_state=bnb_state,
                absmax=None,
                out=None,
                blocksize=blocksize,
                quant_type=quant_type,
            ),
            tag="Baseline",
            metadata=metadata,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            device=device,
        )
    else:
        print("Skipping checking vs. reference bitsandbytes implementation...", file=sys.stderr)
        baseline_result = None

    triton_result = benchmark_it(
        lambda: triton_dequantize_4bit(
            triton_quantized,
            quant_state=triton_state,
            absmax=None,
            out=None,
            blocksize=blocksize,
            quant_type=quant_type,
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
    if baseline_result is not None:
        baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
