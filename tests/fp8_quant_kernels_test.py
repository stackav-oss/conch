# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton fp8 quant kernels."""

from typing import Final

import pytest
import torch

from conch.ops.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_triton
from conch.platforms import current_platform
from conch.reference.quantization.fp8 import scaled_fp8_quant as scaled_fp8_quant_reference
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.half, torch.bfloat16, torch.float]
_HIDDEN_SIZES: Final = [16, 67, 768, 5137, 8193]
_NUM_TOKENS: Final = [1, 7, 83, 4096]
_SEEDS: Final = [0]
_SCALE: Final = [0.1, 2.1]


def _dequantize(quantized_tensor: torch.Tensor, inv_scale: float, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a quantized tensor."""
    return quantized_tensor.to(dtype) * inv_scale


# Skip this test case if FP8 not supported
@pytest.mark.skipif(not current_platform.supports_fp8(), reason="FP8 is not supported on this GPU type.")
@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("scale", _SCALE)
@torch.inference_mode()
def test_static_scaled_fp8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    scale: float,
) -> None:
    """Test Triton static_scaled_fp8_quant vs. reference implementation."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device)
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    triton_output, _ = scaled_fp8_quant_triton(x, scale_arg)
    reference_output = scaled_fp8_quant_reference(x, scale_arg)

    # Note: Torch doesn't have enough support for fp8 to compare tensors, so we need to dequantize them
    # before we can compare
    torch.testing.assert_close(_dequantize(triton_output, scale, dtype), _dequantize(reference_output, scale, dtype))
