# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of FP8 quantization kernels."""

import torch

from conch import envs
from conch.platforms import current_platform


def _scaled_fp8_quant_pytorch_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """PyTorch reference fp8 quant impl."""
    quant_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn
    finfo = torch.finfo(quant_dtype)
    scale = scale.reciprocal()
    qweight = (x.to(torch.float32) * scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(quant_dtype)


def _scaled_fp8_quant_vllm_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """vLLM reference fp8 quant impl."""
    from vllm._custom_ops import scaled_fp8_quant as scaled_fp8_quant_vllm

    output, _ = scaled_fp8_quant_vllm(x, scale)
    return output  # type: ignore[no-any-return, unused-ignore]


def scaled_fp8_quant(
    input_tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled fp8 quantization (currently only static supported)."""
    if scale is None:
        error_msg = "Dynamic quantization not implemented"
        raise NotImplementedError(error_msg)

    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _scaled_fp8_quant_vllm_ref(input_tensor, scale)

    return _scaled_fp8_quant_pytorch_ref(input_tensor, scale)
