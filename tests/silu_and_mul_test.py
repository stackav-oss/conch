# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test for silu_and_mul Triton port."""

from typing import Final

import pytest
import torch

from conch.ops.activation.silu_and_mul import silu_and_mul as silu_and_mul_triton
from conch.platforms import current_platform
from conch.reference.activation.silu_and_mul import silu_and_mul as silu_and_mul_reference
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.float32, torch.float16, torch.bfloat16]
_DIM: Final = [855, 2048]
_BATCH_SIZE: Final = [4, 10, 32]
_NUM_TOKENS: Final = [20, 512]
_USE_BATCH_SIZE: Final = [True, False]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("d", _DIM)
@pytest.mark.parametrize("batch_size", _BATCH_SIZE)
@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("use_batch_size", _USE_BATCH_SIZE)
def test_silu_and_mul(dtype: torch.dtype, d: int, batch_size: int, num_tokens: int, use_batch_size: bool) -> None:
    """Test Triton SiLU implementation."""
    seed_everything(0)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    x_shape = (batch_size, num_tokens, 2 * d) if use_batch_size else (num_tokens, 2 * d)
    x = torch.randn(x_shape, dtype=dtype, device=device)

    ref_output = silu_and_mul_reference(x)
    triton_output = silu_and_mul_triton(x)

    tolerance = 1e-3
    torch.testing.assert_close(triton_output, ref_output, rtol=tolerance, atol=tolerance)
