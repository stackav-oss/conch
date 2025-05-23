# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of RMS norm kernel."""

import torch

from conch import envs
from conch.platforms import current_platform


def _rms_norm_pytorch_ref(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """PyTorch reference rms_norm impl."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    x = x.to(orig_dtype) * weight
    return x


def _rms_norm_vllm_ref(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """vLLM reference rms_norm impl."""
    from vllm._custom_ops import rms_norm as rms_norm_cuda

    out = torch.empty_like(x, dtype=x.dtype, device=x.device)
    rms_norm_cuda(out, x, weight, epsilon)

    return out


def _fused_add_rms_norm_pytorch_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference fused_add_rms_norm impl."""
    x = x + residual
    residual = x

    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    x = x.to(orig_dtype) * weight
    return x, residual


def _fused_add_rms_norm_vllm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM reference fused_add_rms_norm impl."""
    from vllm._custom_ops import fused_add_rms_norm as fused_add_rms_norm_cuda

    fused_add_rms_norm_cuda(x, residual, weight, epsilon)

    return x, residual  # type: ignore[no-any-return, unused-ignore]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """RMS norm operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _rms_norm_vllm_ref(
            x,
            weight,
            epsilon,
        )

    return _rms_norm_pytorch_ref(
        x,
        weight,
        epsilon,
    )


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused add RMS norm operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _fused_add_rms_norm_vllm_ref(
            x,
            residual,
            weight,
            epsilon,
        )

    return _fused_add_rms_norm_pytorch_ref(
        x,
        residual,
        weight,
        epsilon,
    )
