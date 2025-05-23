# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of silu and mul kernel."""

import torch
import torch.nn.functional as F  # noqa: N812

from conch import envs
from conch.platforms import current_platform


def _silu_and_mul_pytorch_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference silu and mul implementation."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _silu_and_mul_vllm_ref(x: torch.Tensor) -> torch.Tensor:
    """vLLM reference silu and mul implementation."""
    from vllm.model_executor.layers.activation import SiluAndMul

    silu_layer = SiluAndMul()  # type: ignore[no-untyped-call, unused-ignore]
    return silu_layer.forward_cuda(x)  # type: ignore[no-any-return, unused-ignore]


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Silu and mul operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _silu_and_mul_vllm_ref(x)

    return _silu_and_mul_pytorch_ref(x)
