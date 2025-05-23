# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Bitsandbytes dequantize PyTorch reference implementation."""

from collections.abc import Callable

import torch
import triton


def _pytorch_fp4_dequantize(x: torch.Tensor, code: torch.Tensor | None) -> float:  # noqa: C901, PLR0911, PLR0912
    """Dequantize an FP4 value to floating point."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is None

    if x == 15:
        return -0.25
    if x == 14:
        return -0.166666
    if x == 13:
        return -0.5
    if x == 12:
        return -0.333333
    if x == 11:
        return -1.0
    if x == 10:
        return -0.666666
    if x == 9:
        return -0.0052083333
    if x == 8:
        return -0.0
    if x == 7:
        return 0.25
    if x == 6:
        return 0.166666
    if x == 5:
        return 0.5
    if x == 4:
        return 0.333333
    if x == 3:
        return 1.0
    if x == 2:
        return 0.6666666
    if x == 1:
        return 0.0052083333
    return 0.0


def _pytorch_nf4_dequantize(x: torch.Tensor, code: torch.Tensor | None) -> float:  # noqa: C901, PLR0911, PLR0912
    """Dequantize an NF4 value to floating point."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is None
    if x == 15:
        return 1.0
    if x == 14:
        return 0.7229568362236023
    if x == 13:
        return 0.5626170039176941
    if x == 12:
        return 0.44070982933044434
    if x == 11:
        return 0.33791524171829224
    if x == 10:
        return 0.24611230194568634
    if x == 9:
        return 0.16093020141124725
    if x == 8:
        return 0.07958029955625534
    if x == 7:
        return 0.0
    if x == 6:
        return -0.09105003625154495
    if x == 5:
        return -0.18477343022823334
    if x == 4:
        return -0.28444138169288635
    if x == 3:
        return -0.39491748809814453
    if x == 2:
        return -0.5250730514526367
    if x == 1:
        return -0.6961928009986877
    return -1.0


def _pytorch_fp8_dequantize(x: torch.Tensor, code: torch.Tensor | None) -> float:
    """Dequantize an FP8 value to fp32."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is not None
    return code[int(x.item())].item()


def _get_pytorch_dequant_method(quant_type: str) -> Callable[[torch.Tensor, torch.Tensor | None], float]:
    """Get PyTorch dequantization method."""
    if quant_type == "nf4":
        return _pytorch_nf4_dequantize
    if quant_type == "fp4":
        return _pytorch_fp4_dequantize
    return _pytorch_fp8_dequantize


def dequantize_blockwise_launcher(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    code: torch.Tensor | None,
    blocksize: int,
    output_size: int,
    quant_type: str,
) -> torch.Tensor:
    """Convert FP32/FP16/BF16 input tensor to NF4/FP4/FP8."""
    assert x.dtype == torch.uint8

    dequant_method = _get_pytorch_dequant_method(quant_type)

    num_blocks = triton.cdiv(output_size, blocksize)
    input_blocksize = blocksize // 2 if quant_type in ("fp4", "nf4") else blocksize

    for block_index in range(num_blocks):
        input_block_offset = block_index * input_blocksize
        output_block_offset = block_index * blocksize

        local_absmax = absmax[block_index]

        this_output_block_size = min(blocksize, output_size - output_block_offset)
        this_input_block_size = this_output_block_size // 2 if quant_type in ("fp4", "nf4") else this_output_block_size

        for element_index in range(this_input_block_size):
            if quant_type in ("fp4", "nf4"):
                packed = x[input_block_offset + element_index]
                x1 = dequant_method(packed >> 4, code) * local_absmax
                x2 = dequant_method(packed & 0x0F, code) * local_absmax

                output_element_offset = element_index * 2
                out[output_block_offset + output_element_offset] = x1
                out[output_block_offset + output_element_offset + 1] = x2
            else:
                result = dequant_method(x[input_block_offset + element_index], code) * local_absmax
                out[output_block_offset + element_index] = result

    return out
