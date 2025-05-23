# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton rms_norm."""

from typing import Final

import pytest
import torch

from conch.ops.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_triton
from conch.ops.normalization.rms_norm import rms_norm as rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.rms_norm import fused_add_rms_norm as fused_add_rms_norm_reference
from conch.reference.normalization.rms_norm import rms_norm as rms_norm_reference
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.float16, torch.bfloat16, torch.float32]
_HIDDEN_SIZES: Final = [855, 1024]
_NUM_TOKENS: Final = [80, 256]
_EPSILONS: Final = [1e-6, 1e-4]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("epsilon", _EPSILONS)
def test_rms_norm(dtype: torch.dtype, hidden_size: int, num_tokens: int, epsilon: float) -> None:
    """Test Triton rms_norm implementation."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    x_shape = (num_tokens, hidden_size)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    triton_output = rms_norm_triton(x, weight, epsilon)
    reference_output = rms_norm_reference(x, weight, epsilon)

    tolerance = 1e-2
    torch.testing.assert_close(triton_output, reference_output, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("epsilon", _EPSILONS)
def test_fused_add_rms_norm(dtype: torch.dtype, hidden_size: int, num_tokens: int, epsilon: float) -> None:
    """Test Triton fused_add_rms_norm implementation."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    x_shape = (num_tokens, hidden_size)
    x = torch.randn(x_shape, dtype=dtype, device=device)
    residual = torch.randn(x_shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)

    # We need to clone the tensors because the fused_add_rms_norm kernel modifies them in-place
    triton_x = x.clone()
    reference_x = x.clone()
    triton_residual = residual.clone()
    reference_residual = residual.clone()

    triton_output, triton_residual = fused_add_rms_norm_triton(triton_x, triton_residual, weight, epsilon)
    reference_output, reference_residual = fused_add_rms_norm_reference(
        reference_x, reference_residual, weight, epsilon
    )

    tolerance = 1e-2
    torch.testing.assert_close(triton_output, reference_output, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(triton_residual, reference_residual, rtol=tolerance, atol=tolerance)
