# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of gelu tanh and mul kernel."""

import torch
import torch.nn.functional as F  # noqa: N812

from conch import envs
from conch.platforms import current_platform


def _gelu_tanh_and_mul_pytorch_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference gelu_tanh_and_mul impl."""
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]


def _gelu_tanh_and_mul_vllm_ref(x: torch.Tensor) -> torch.Tensor:
    """vLLM reference gelu_tanh_and_mul impl."""
    from vllm.model_executor.layers.activation import GeluAndMul

    gelu_layer = GeluAndMul("tanh")
    return gelu_layer.forward_cuda(x)  # type: ignore[no-any-return, unused-ignore]


def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Gelu, tanh, and mul operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _gelu_tanh_and_mul_vllm_ref(x)

    return _gelu_tanh_and_mul_pytorch_ref(x)
