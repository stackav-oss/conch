# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for scaled GEMM."""

from typing import Final

import pytest
import torch

from conch.ops.quantization.gemm import scaled_gemm as scaled_gemm_triton
from conch.platforms import current_platform
from conch.reference.quantization.scaled_gemm import scaled_gemm as scaled_gemm_reference
from conch.third_party.vllm.utils import seed_everything


def _is_floating_point_type(dtype: torch.dtype) -> bool:
    """Check whether a type is floating point."""
    return torch.tensor([1, 1], dtype=dtype).is_floating_point()


def _get_8bit_types() -> list[torch.dtype]:
    types = [torch.int8]
    if current_platform.is_amd() and current_platform.supports_fp8():
        types.append(torch.float8_e4m3fnuz)
    elif current_platform.is_nvidia() and current_platform.supports_fp8():
        types.append(torch.float8_e4m3fn)
    return types


@pytest.mark.parametrize(("m_dim", "k_dim", "n_dim"), [(128, 256, 128), (1024, 1024, 1024), (4096, 2048, 4096)])
@pytest.mark.parametrize("input_dtype", _get_8bit_types())
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_scalar_scale_a", [True, False])
@pytest.mark.parametrize("use_scalar_scale_b", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_scaled_gemm(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    use_scalar_scale_a: bool,
    use_scalar_scale_b: bool,
    use_bias: bool,
) -> None:
    """Test mixed precision GEMM Triton kernel."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    if use_scalar_scale_a:
        scale_a_tensor = torch.rand((1, 1), dtype=torch.float32, device=device)
    else:
        scale_a_tensor = 0.25 * torch.rand((m_dim, 1), device=device)

    if use_scalar_scale_b:
        scale_b_tensor = torch.rand((1, 1), dtype=torch.float32, device=device)
    else:
        scale_b_tensor = 0.25 * torch.rand((n_dim, 1), device=device)

    if _is_floating_point_type(input_dtype):
        a = (0.25 * torch.rand((m_dim, k_dim), dtype=torch.float32, device=device)).to(input_dtype)
        b = (0.25 * torch.rand((n_dim, k_dim), dtype=torch.float32, device=device)).to(input_dtype).T
    else:
        a = torch.randint(-32, 32, (m_dim, k_dim), dtype=input_dtype, device=device)
        b = torch.randint(-32, 32, (n_dim, k_dim), dtype=input_dtype, device=device).T

    bias = None
    if use_bias:
        bias = torch.rand((n_dim,), device=device, dtype=output_dtype)

    reference_output = scaled_gemm_reference(a, b, scale_a_tensor, scale_b_tensor, output_dtype, bias)
    triton_output = scaled_gemm_triton(a, b, scale_a_tensor, scale_b_tensor, output_dtype, bias)

    torch.testing.assert_close(reference_output, triton_output, rtol=1e-1, atol=1e-1)
