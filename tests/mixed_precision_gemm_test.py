# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for mixed precision GEMM."""

import math
from typing import Final

import pytest
import torch

from conch.ops.quantization.gemm import mixed_precision_gemm
from conch.platforms import current_platform
from conch.third_party.vllm.quant_utils import pack_rows, quantize_weights
from conch.third_party.vllm.scalar_type import ScalarType, scalar_types
from conch.third_party.vllm.utils import seed_everything


def _quantize_and_pack(
    w: torch.Tensor,
    wtype: ScalarType,
    group_size: int,
    use_zero_points: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Quantize and pack."""

    w_ref, w_q, w_s, w_zp = quantize_weights(
        w, wtype, group_size, zero_points=use_zero_points, ref_zero_points_after_scales=False
    )

    w_q_packed = pack_rows(w_q, wtype.size_bits, *w_q.shape)

    return w_ref, w_q_packed, w_s, w_zp


@pytest.mark.parametrize(("m_dim", "k_dim", "n_dim"), [(128, 256, 128), (1024, 1024, 1024), (4096, 2048, 4096)])
@pytest.mark.parametrize(
    "weight_dtype", [scalar_types.uint4b8, scalar_types.uint8b128, scalar_types.uint4, scalar_types.uint8]
)
@pytest.mark.parametrize("use_zero_points", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_gemm(
    m_dim: int,
    k_dim: int,
    n_dim: int,
    weight_dtype: ScalarType,
    use_zero_points: bool,
    input_dtype: torch.dtype,
) -> None:
    """Test mixed precision GEMM Triton kernel."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    group_size: Final = 128
    assert group_size <= k_dim

    a = (10 * (torch.rand((m_dim, k_dim), dtype=torch.float32, device=device) - 0.3)).to(input_dtype)
    b = (10 * (torch.rand((k_dim, n_dim), dtype=torch.float32, device=device) - 0.3)).to(input_dtype)

    w_ref, w_q_packed, w_s, w_zp = _quantize_and_pack(b, weight_dtype, group_size, use_zero_points)

    # For mypy
    if use_zero_points:
        assert w_zp is not None
    assert w_s is not None

    output_ref = torch.matmul(a, w_ref)

    triton_output = mixed_precision_gemm(a, w_q_packed, w_s, w_zp, weight_dtype, group_size)

    atol = min(5e-2 * math.sqrt(k_dim), 1)
    torch.testing.assert_close(triton_output, output_ref, rtol=1e-1, atol=atol)
