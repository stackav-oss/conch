# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Scaled matrix multiplication kernel benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.quantization.gemm import scaled_gemm as scaled_gemm_triton
from conch.platforms import current_platform
from conch.reference.quantization.scaled_gemm import scaled_gemm as scaled_gemm_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map click arg for dtype to torch type."""
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "int8":
        return torch.int8
    if dtype_str == "fp8":
        return torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn

    error_msg = f"Unrecognized data type: '{dtype_str}'"
    raise ValueError(error_msg)


def _is_floating_point_type(dtype: torch.dtype) -> bool:
    """Check whether a type is floating point."""
    return torch.tensor([1, 1], dtype=dtype).is_floating_point()


@click.command()
@click.option(
    "--m-dim",
    required=True,
    type=int,
    default=4096,
    help="1st dimension of A matrix (M x K)",
)
@click.option(
    "--k-dim",
    required=True,
    type=int,
    default=8192,
    help="Common dimension of A and B matrices (M x K) * (K * N)",
)
@click.option(
    "--n-dim",
    required=True,
    type=int,
    default=4096,
    help="2nd dimension of B matrix (K x N)",
)
@click.option(
    "--scale-a",
    required=True,
    type=float,
    default=2.1,
    help="Scaling arg for first input",
)
@click.option(
    "--scale-b",
    required=True,
    type=float,
    default=1.2,
    help="Scaling arg for second input",
)
@click.option(
    "--input-dtype",
    required=True,
    type=click.Choice(["fp8", "int8"]),
    default="int8",
    help="Data type of input",
)
@click.option(
    "--output-dtype",
    required=True,
    type=click.Choice(["fp16", "bf16"]),
    default="bf16",
    help="Data type of output",
)
@click.option(
    "--use-bias",
    is_flag=True,
    help="Flag for adding a bias",
)
@click.option(
    "--use-scalar-scale-a",
    is_flag=True,
    help="Flag for using scalar or vector for scale_a",
)
@click.option(
    "--use-scalar-scale-b",
    is_flag=True,
    help="Flag for using scalar or vector for scale_b",
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
    m_dim: int,
    k_dim: int,
    n_dim: int,
    scale_a: float,
    scale_b: float,
    input_dtype: str,
    output_dtype: str,
    use_bias: bool,
    use_scalar_scale_a: bool,
    use_scalar_scale_b: bool,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton scaled matrix mulpity.

    Args:
        m_dim: 1st dimension of matrix A.
        k_dim: Common dimension between matrix A and B.
        n_dim: 2nd dimension of matrix B.
        scale_a: Scaling factor to apply to matrix A.
        scale_b: Scaling factor to apply to matrix B.
        input_dtype: Data type of input matrices.
        output_dtype: Data type of output matrices.
        use_bias: Whether or not to use bias.
        use_scalar_scale_a: Whether or not to use scalar scale_a.
        use_scalar_scale_b: Whether or not to use scalar scale_b.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
    """
    if input_dtype == "fp8" and not current_platform.supports_fp8():
        error_msg = "FP8 not supported on this GPU, cannot run benchmark!"
        raise NotImplementedError(error_msg)

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    input_dtype_torch: Final = _to_torch_dtype(input_dtype)
    output_dtype_torch: Final = _to_torch_dtype(output_dtype)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "m_dim": m_dim,
            "k_dim": k_dim,
            "n_dim": n_dim,
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
        },
    )

    if use_scalar_scale_a:
        scale_a_tensor = torch.tensor(scale_a, dtype=torch.float32, device=device)
    else:
        scale_a_tensor = 0.25 * torch.rand((m_dim, 1), device=device)

    if use_scalar_scale_b:
        scale_b_tensor = torch.tensor(scale_b, dtype=torch.float32, device=device)
    else:
        scale_b_tensor = 0.25 * torch.rand((n_dim, 1), device=device)

    if _is_floating_point_type(input_dtype_torch):
        a = (0.25 * torch.rand((m_dim, k_dim), dtype=torch.float32, device=device)).to(input_dtype_torch)
        b = (0.25 * torch.rand((n_dim, k_dim), dtype=torch.float32, device=device)).to(input_dtype_torch).T
    else:
        a = torch.randint(-32, 32, (m_dim, k_dim), dtype=input_dtype_torch, device=device)
        b = torch.randint(-32, 32, (n_dim, k_dim), dtype=input_dtype_torch, device=device).T

    bias = None
    if use_bias:
        bias = torch.rand((n_dim,), device=device, dtype=output_dtype_torch)

    reference_output = scaled_gemm_reference(a, b, scale_a_tensor, scale_b_tensor, output_dtype_torch, bias)
    triton_output = scaled_gemm_triton(a, b, scale_a_tensor, scale_b_tensor, output_dtype_torch, bias)

    if not torch.allclose(reference_output, triton_output, atol=1e-1, rtol=1e-1):
        print("WARNING: Reference and Triton results differ!", file=sys.stderr)
        print(f"Output max diff: {(reference_output - triton_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {reference_output}", file=sys.stderr)
            print(f"Triton output: {triton_output}", file=sys.stderr)
    else:
        print("Results matched :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: scaled_gemm_reference(
            a,
            b,
            scale_a_tensor,
            scale_b_tensor,
            output_dtype_torch,
            bias,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    triton_result = benchmark_it(
        lambda: scaled_gemm_triton(
            a,
            b,
            scale_a_tensor,
            scale_b_tensor,
            output_dtype_torch,
            bias,
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
