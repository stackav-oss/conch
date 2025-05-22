# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of int8 quantization kernels."""

import torch

from conch import envs
from conch.platforms import current_platform


def _scaled_int8_quant_pytorch_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """PyTorch reference int8 quant impl."""
    quant_dtype = torch.int8
    iinfo = torch.iinfo(quant_dtype)
    scale = scale.reciprocal()
    qweight = (x * scale).clamp(min=iinfo.min, max=iinfo.max)
    return qweight.to(quant_dtype)


def _scaled_int8_quant_vllm_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """vLLM reference int8 quant impl."""
    from vllm._custom_ops import scaled_int8_quant as scaled_int8_quant_vllm

    output, _, _ = scaled_int8_quant_vllm(x, scale)
    return output  # type: ignore[no-any-return, unused-ignore]


def scaled_int8_quant(
    input_tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled int8 quantization (currently only static supported)."""
    if scale is None:
        error_msg = "Dynamic quantization not implemented"
        raise NotImplementedError(error_msg)

    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _scaled_int8_quant_vllm_ref(input_tensor, scale)

    return _scaled_int8_quant_pytorch_ref(input_tensor, scale)
