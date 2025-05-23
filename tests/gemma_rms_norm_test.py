# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test Triton port of GemmaRMSNorm."""

from typing import Final

import pytest
import torch

from conch.ops.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_triton
from conch.platforms import current_platform
from conch.reference.normalization.gemma_rms_norm import gemma_rms_norm as gemma_rms_norm_reference
from conch.third_party.vllm.utils import seed_everything

_N_POS: Final = [2, 500, 1000]
_HIDDEN_SIZES: Final = [1024, 2048, 4096]


@pytest.mark.parametrize("n_pos", _N_POS)
@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
@torch.inference_mode()
def test_kernel(
    n_pos: int,
    hidden_size: int,
) -> None:
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    epsilon: Final = 1e-6

    x = torch.randn((n_pos, hidden_size), dtype=torch.float16, device=device)
    weights = torch.randn((hidden_size,), device=device)

    x_ref = x.clone()
    x_triton = x.clone()

    result_ref = gemma_rms_norm_reference(x_ref, weights, epsilon, residual=None)
    result_triton = gemma_rms_norm_triton(x_triton, weights, epsilon, residual=None)

    torch.testing.assert_close(result_ref, result_triton, atol=1e-5, rtol=0.001)
