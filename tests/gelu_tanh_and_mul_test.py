# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test for gelu_tanh_and_mul Triton port."""

from typing import Final

import pytest
import torch

from conch.ops.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_triton
from conch.platforms import current_platform
from conch.reference.activation.gelu_tanh_and_mul import gelu_tanh_and_mul as gelu_tanh_and_mul_reference
from conch.third_party.vllm.utils import seed_everything

_NUM_TOKENS: Final = [128, 2048]
_DIM: Final = [2048, 13824]


@pytest.mark.parametrize("n_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("d", _DIM)
@torch.inference_mode()
def test_kernel(n_tokens: int, d: int) -> None:
    seed_everything(0)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    projections = torch.rand((n_tokens, d * 2), device=device)

    ref_output = gelu_tanh_and_mul_reference(projections)
    triton_output = gelu_tanh_and_mul_triton(projections)

    torch.testing.assert_close(triton_output, ref_output)
