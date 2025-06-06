# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of Gemma RMS norm kernel."""

import torch

from conch import envs
from conch.platforms import current_platform


def _gemma_rms_norm_pytorch_ref(
    weight: torch.Tensor,
    variance_epsilon: float,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference gemma_rms_norm impl."""
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x

    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + variance_epsilon)
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    x = x * (1.0 + weight.float())
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)


def _gemma_rms_norm_vllm_ref(
    weight: torch.Tensor,
    variance_epsilon: float,
    x: torch.Tensor,
    residual: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """vLLM reference gemma_rms_norm impl."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    layer = GemmaRMSNorm(hidden_size=weight.size(0), eps=variance_epsilon)
    layer.weight = torch.nn.Parameter(weight)

    return layer.forward_cuda(x, residual)  # type: ignore[no-any-return, unused-ignore]


def gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMS norm operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _gemma_rms_norm_vllm_ref(
            weight,
            variance_epsilon,
            x,
            residual,
        )

    return _gemma_rms_norm_pytorch_ref(
        weight,
        variance_epsilon,
        x,
        residual,
    )
