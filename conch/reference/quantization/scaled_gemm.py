# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of scaled GEMM."""

import torch

from conch import envs
from conch.platforms import current_platform


def _scaled_gemm_pytorch_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    out = scale_a * out
    out = scale_b.T * out
    out = out.to(out_dtype)
    if bias is not None:
        out = out + bias

    return out


def _scaled_gemm_vllm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm._custom_ops import cutlass_scaled_mm

    return cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)  # type: ignore[no-any-return, unused-ignore]


def scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # CUTLASS is only supported on Nvidia
    if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
        return _scaled_gemm_vllm_ref(
            a,
            b,
            scale_a,
            scale_b,
            out_dtype,
            bias,
        )

    return _scaled_gemm_pytorch_ref(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype,
        bias,
    )
