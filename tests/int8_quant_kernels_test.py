# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton int8 quant kernels."""

from typing import Final

import pytest
import torch

from conch.ops.quantization.int8 import scaled_int8_quant as scaled_int8_quant_triton
from conch.platforms import current_platform
from conch.reference.quantization.int8 import scaled_int8_quant as scaled_int8_quant_reference
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.half, torch.bfloat16, torch.float]
_HIDDEN_SIZES: Final = [16, 67, 768, 5137, 8193]
_NUM_TOKENS: Final = [1, 7, 83, 4096]
_SEEDS: Final = [0]
_SCALE: Final = [0.1, 2.1]


@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("scale", _SCALE)
@torch.inference_mode()
def test_static_scaled_int8_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
    scale: float,
) -> None:
    """Test Triton static_scaled_int8_quant vs. reference implementation."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device=device) * 1000
    scale_arg = torch.tensor([scale], dtype=torch.float32, device=device)

    triton_output, _ = scaled_int8_quant_triton(x, scale_arg)

    ref_output = scaled_int8_quant_reference(x, scale_arg)

    # big atol to account for rounding errors
    torch.testing.assert_close(triton_output, ref_output, atol=1, rtol=0.0)
