# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

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
    from vllm._custom_ops import silu_and_mul as silu_and_mul_cuda

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    silu_and_mul_cuda(out, x)

    return out


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Silu and mul operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _silu_and_mul_vllm_ref(x)

    return _silu_and_mul_pytorch_ref(x)
