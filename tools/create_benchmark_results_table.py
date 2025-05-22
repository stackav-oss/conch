# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Create markdown table for README."""

import os
from pathlib import Path
from subprocess import run
from typing import Final

import click
import pandas as pd  # type: ignore[import-untyped]

from conch.platforms import current_platform

_MARKDOWN_TABLE_HEADER: Final = """\
| Operation | CUDA Runtime | Triton Runtime | Triton Speedup |
| --- | --- | --- | --- |
"""

_MARKDOWN_TABLE_ROW: Final = "| {} | {:.3f} ms | {:.3f} ms | {:.2f} |\n"

_TABLE_OP_NAME_TO_BENCHMARK: Final = {
    "GeLU, Tanh, and Mul": "gelu_tanh_and_mul_benchmark",
    "SiLU and Mul": "silu_and_mul_benchmark",
    "Paged Attention": "paged_attention_vs_flash_benchmark",
    "Rotary Embedding": "rotary_embedding_benchmark",
    "RMS Norm (Gemma-style)": "gemma_rms_norm_benchmark",
    "RMS Norm (Llama-style)": "rms_norm_benchmark",
    "bitsandbytes: Dequantize": "bnb_dequantize_blockwise_benchmark",
    "bitsandbytes: Quantize": "bnb_quantize_blockwise_benchmark",
    "FP8 Static Quantization": "static_scaled_fp8_quant_benchmark",
    "Int8 Static Quantization": "static_scaled_int8_quant_benchmark",
    "Mixed-precision GEMM [Int4 x FP16]": "mixed_precision_gemm_benchmark",
    "Scaled GEMM [Int8 x BF16]": "scaled_gemm_benchmark",
    "vLLM: Copy Blocks": "copy_blocks_benchmark",
    "vLLM: Reshape and Cache": "reshape_and_cache_benchmark",
}

_DEVICE_SPECIFIC_BLACKLIST: Final = {
    "NVIDIA A10": [
        "static_scaled_fp8_quant_benchmark",
        "mixed_precision_gemm_benchmark",
    ],
    "unknown": [],
}


@click.command()
@click.option(
    "--results-directory",
    required=True,
    type=click.Path(path_type=Path, dir_okay=True, file_okay=False),
    help="Path to the benchmark results directory.",
)
@click.option(
    "--use-cached-results",
    is_flag=True,
    help="Flag to use cached benchmark results if they exist",
)
def main(results_directory: Path, use_cached_results: bool) -> None:
    """Main function to plot benchmarking results."""
    # Always run against fastest possible implementation
    os.environ["CONCH_BENCH_ENABLE_ALL_REF"] = "1"
    os.environ["CONCH_ENABLE_BNB"] = "1"
    os.environ["CONCH_ENABLE_VLLM"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"

    # Create directory for output if it doesn't exist already
    results_directory.mkdir(parents=True, exist_ok=True)

    # Running string for markdown table output
    result = _MARKDOWN_TABLE_HEADER

    device_name = current_platform.get_device_name()

    for op_name, benchmark_name in _TABLE_OP_NAME_TO_BENCHMARK.items():
        if (
            device_name in _DEVICE_SPECIFIC_BLACKLIST.keys()
            and benchmark_name in _DEVICE_SPECIFIC_BLACKLIST[device_name]
        ):
            print(f"Skipping {op_name} benchmark because it is blacklisted on current platform ({device_name = })")
            continue

        results_csv = results_directory / f"{benchmark_name}.csv"

        if use_cached_results and results_csv.exists():
            print(f"Skipping {op_name} benchmark, CSV file already exists.")
        else:
            # Run benchmark and redirect output
            print(f"Running benchmark for {op_name}...")

            with results_csv.open("w") as results_file:
                run(
                    ["python", f"benchmarks/{benchmark_name}.py", "--csv", "--num-iterations", "10000"],
                    check=True,
                    stdout=results_file,
                )

        # Read the CSV file
        df = pd.read_csv(results_csv)
        triton_df = df[df["tag"] == "Triton"]
        baseline_df = df[df["tag"] == "Baseline"]

        # Calculate speedup
        triton_runtime = triton_df["runtime_ms"].iloc[0]
        baseline_runtime = baseline_df["runtime_ms"].iloc[0]
        speedup = baseline_runtime / triton_runtime

        # Create a markdown table row
        row = _MARKDOWN_TABLE_ROW.format(
            op_name,
            baseline_runtime,
            triton_runtime,
            speedup,
        )
        result += row

    # Print markdown table
    print("\nResults:\n")
    print(result)


if __name__ == "__main__":
    main()
