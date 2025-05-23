# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-precision matrix multiplication kernel benchmark."""

import math
import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.quantization.gemm import mixed_precision_gemm
from conch.platforms import current_platform
from conch.third_party.vllm.quant_utils import pack_rows, quantize_weights
from conch.third_party.vllm.scalar_type import ScalarType, scalar_types
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
    from vllm import _custom_ops as vllm_custom_ops
else:
    vllm_custom_ops = None  # type: ignore[assignment, unused-ignore]


def _to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Map click arg for dtype to torch type."""
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32

    error_msg = f"Unrecognized data type: '{dtype_str}'"
    raise ValueError(error_msg)


def _to_scalar_dtype(dtype_str: str) -> ScalarType:
    """Map click arg for dtype to scalar type."""
    if dtype_str == "uint4":
        return scalar_types.uint4
    if dtype_str == "uint8":
        return scalar_types.uint8
    if dtype_str == "uint4b8":
        return scalar_types.uint4b8
    if dtype_str == "uint8b128":
        return scalar_types.uint8b128

    error_msg = f"Unrecognized data type: '{dtype_str}'"
    raise ValueError(error_msg)


def _machete_quantize_and_pack(
    w: torch.Tensor, wtype: ScalarType, group_size: int, enable_machete: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and pack weight matrix."""
    w_ref, w_q, w_s, _ = quantize_weights(
        w,
        wtype,
        group_size,
        zero_points=False,
    )

    w_q_packed = pack_rows(w_q, wtype.size_bits, *w_q.shape)

    w_q_machete = w_q_packed.t().contiguous().t()
    if enable_machete and vllm_custom_ops is not None:
        w_q_machete = vllm_custom_ops.machete_prepack_B(
            w_q_machete, a_type=w.dtype, b_type=wtype, group_scales_type=w.dtype
        )

    return w_ref, w_q_machete, w_q_packed, w_s


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
    "--input-dtype",
    required=True,
    type=click.Choice(["fp16", "bf16", "fp32"]),
    default="fp16",
    help="Data type of input",
)
@click.option(
    "--weight-dtype",
    required=True,
    type=click.Choice(["uint4", "uint8", "uint4b8", "uint8b128"]),
    default="uint4b8",
    help="Data type of input",
)
@click.option(
    "--enable-machete",
    is_flag=True,
    default=envs.CONCH_BENCH_ENABLE_ALL_REF,
    help="Flag for enabling running Machete (only on H100)",
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
    input_dtype: str,
    weight_dtype: str,
    enable_machete: bool,
    num_iterations: int,
    num_warmup_iterations: int,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark mixed-precision matrix multiply.

    Args:
        m_dim: 1st dimension of matrix A.
        k_dim: Common dimension between matrix A and B.
        n_dim: 2nd dimension of matrix B.
        input_dtype: Data type of input matrices.
        weight_dtype: Data type of weight matrices.
        enable_machete: Enable running Machete kernel.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    input_dtype_torch: Final = _to_torch_dtype(input_dtype)
    weight_dtype_vllm: Final = _to_scalar_dtype(weight_dtype)

    group_size: Final = 128
    assert group_size <= k_dim

    if enable_machete and vllm_custom_ops is None:
        error_msg = "In order to enable machete baseline we vLLM must be enabled via `CONCH_ENABLE_VLLM=1`."
        raise ValueError(error_msg)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "m_dim": m_dim,
            "k_dim": k_dim,
            "n_dim": n_dim,
            "input_dtype": input_dtype,
            "weight_dtype": weight_dtype,
        },
    )

    a = (10 * (torch.rand((m_dim, k_dim), dtype=torch.float32, device=device) - 0.3)).to(input_dtype_torch)
    b = (10 * (torch.rand((k_dim, n_dim), dtype=torch.float32, device=device) - 0.3)).to(input_dtype_torch)

    w_ref, w_q_machete, w_q, w_s = _machete_quantize_and_pack(b, weight_dtype_vllm, group_size, enable_machete)

    output_ref = torch.matmul(a, w_ref)

    triton_output = mixed_precision_gemm(a, w_q, w_s, None, weight_dtype_vllm, group_size)

    # Relax atol as our reduction dim becomes larger (more rounding error)
    atol = min(5e-2 * math.sqrt(k_dim), 1)
    rtol = 1e-1

    if enable_machete and vllm_custom_ops is not None:
        machete_output = vllm_custom_ops.machete_mm(
            a=a,
            b_q=w_q_machete,
            b_type=weight_dtype_vllm,
            b_group_scales=w_s,
            b_group_size=group_size,
        )

        if not torch.allclose(output_ref, machete_output, rtol=rtol, atol=atol):
            print("WARNING: Reference and machete results differ!", file=sys.stderr)
            print(f"Output max diff: {(output_ref - machete_output).abs().max().item()}", file=sys.stderr)

            if verbose:
                print(f"Reference output: {output_ref}", file=sys.stderr)
                print(f"Machete output: {machete_output}", file=sys.stderr)

    if not torch.allclose(output_ref, triton_output, rtol=rtol, atol=atol):
        print("WARNING: Reference and Triton results differ!", file=sys.stderr)
        print(f"Output max diff: {(output_ref - triton_output).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {output_ref}", file=sys.stderr)
            print(f"Triton output: {triton_output}", file=sys.stderr)
    else:
        print("Results matched :)", file=sys.stderr)

    if enable_machete and vllm_custom_ops is not None:
        baseline_result = benchmark_it(
            lambda: vllm_custom_ops.machete_mm(
                a=a,
                b_q=w_q_machete,
                b_type=weight_dtype_vllm,
                b_group_scales=w_s,
                b_group_size=group_size,
            ),
            tag="Baseline",
            metadata=metadata,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            device=device,
        )
    else:
        baseline_result = None

    triton_result = benchmark_it(
        lambda: mixed_precision_gemm(a, w_q, w_s, None, weight_dtype_vllm, group_size),
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
