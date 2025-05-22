# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of bitsandbytes dequantization."""

from enum import Enum

import torch
import triton
import triton.language as tl


class QuantizedDType(Enum):
    """Data types for mapping between Triton/Torch."""

    GENERAL_8BIT = 0
    NF4 = 1
    FP4 = 2


# Triton can only access global values instantiated as tl.constexpr
_QUANTIZED_DTYPE_GENERAL_8BIT: tl.constexpr = tl.constexpr(QuantizedDType.GENERAL_8BIT.value)
_QUANTIZED_DTYPE_NF4: tl.constexpr = tl.constexpr(QuantizedDType.NF4.value)
_QUANTIZED_DTYPE_FP4: tl.constexpr = tl.constexpr(QuantizedDType.FP4.value)


def _get_quantized_dtype(quant_type: str) -> QuantizedDType:
    """Map string quant_type to enum."""
    if quant_type == "nf4":
        return QuantizedDType.NF4
    if quant_type == "fp4":
        return QuantizedDType.FP4
    return QuantizedDType.GENERAL_8BIT


@triton.jit  # type: ignore[misc]
def _nf4_dequantize(x: tl.tensor) -> float:  # noqa: C901, PLR0911, PLR0912
    """Dequantize an NF4 value to floating point."""
    # Note: Triton does not support match/case
    if x == 15:  # noqa: PLR2004
        return 1.0
    if x == 14:  # noqa: PLR2004
        return 0.7229568362236023
    if x == 13:  # noqa: PLR2004
        return 0.5626170039176941
    if x == 12:  # noqa: PLR2004
        return 0.44070982933044434
    if x == 11:  # noqa: PLR2004
        return 0.33791524171829224
    if x == 10:  # noqa: PLR2004
        return 0.24611230194568634
    if x == 9:  # noqa: PLR2004
        return 0.16093020141124725
    if x == 8:  # noqa: PLR2004
        return 0.07958029955625534
    if x == 7:  # noqa: PLR2004
        return 0.0
    if x == 6:  # noqa: PLR2004
        return -0.09105003625154495
    if x == 5:  # noqa: PLR2004
        return -0.18477343022823334
    if x == 4:  # noqa: PLR2004
        return -0.28444138169288635
    if x == 3:  # noqa: PLR2004
        return -0.39491748809814453
    if x == 2:  # noqa: PLR2004
        return -0.5250730514526367
    if x == 1:
        return -0.6961928009986877
    return -1.0


@triton.jit  # type: ignore[misc]
def _fp4_dequantize(x: tl.tensor) -> float:  # noqa: C901, PLR0911, PLR0912
    """Dequantize an FP4 value to floating point."""
    # Note: Triton does not support match/case
    if x == 15:  # noqa: PLR2004
        return -0.25
    if x == 14:  # noqa: PLR2004
        return -0.166666
    if x == 13:  # noqa: PLR2004
        return -0.5
    if x == 12:  # noqa: PLR2004
        return -0.333333
    if x == 11:  # noqa: PLR2004
        return -1.0
    if x == 10:  # noqa: PLR2004
        return -0.666666
    if x == 9:  # noqa: PLR2004
        return -0.0052083333
    if x == 8:  # noqa: PLR2004
        return -0.0
    if x == 7:  # noqa: PLR2004
        return 0.25
    if x == 6:  # noqa: PLR2004
        return 0.166666
    if x == 5:  # noqa: PLR2004
        return 0.5
    if x == 4:  # noqa: PLR2004
        return 0.333333
    if x == 3:  # noqa: PLR2004
        return 1.0
    if x == 2:  # noqa: PLR2004
        return 0.6666666
    if x == 1:
        return 0.0052083333
    return 0.0


@triton.jit  # type: ignore[misc]
def _dequantize_blockwise_kernel(  # noqa: PLR0913
    x_ptr: tl.tensor,
    absmax_ptr: tl.tensor,
    out_ptr: tl.tensor,
    code_ptr: tl.tensor,
    output_size: int,
    cxpr_blocksize: tl.constexpr,
    cxpr_input_blocksize: tl.constexpr,
    cxpr_quantized_dtype: tl.constexpr,
) -> None:
    """Dequantize blockwise kernel."""
    block_index = tl.program_id(0)

    input_block_offset = block_index * cxpr_input_blocksize

    # Get scaling factor for this block
    local_absmax = tl.load(absmax_ptr + block_index)

    # Don't read extra elements if input_size is not perfectly divisible by blocksize
    output_block_offset = block_index * cxpr_blocksize
    this_output_block_size = min(cxpr_blocksize, output_size - output_block_offset)
    this_input_block_size = (
        this_output_block_size if cxpr_quantized_dtype == _QUANTIZED_DTYPE_GENERAL_8BIT else this_output_block_size // 2
    )

    for element_index in range(this_input_block_size):
        if cxpr_quantized_dtype == _QUANTIZED_DTYPE_GENERAL_8BIT:
            x = tl.load(x_ptr + input_block_offset + element_index).to(tl.uint32)
            result = tl.load(code_ptr + x).to(tl.float32) * local_absmax
            tl.store(out_ptr + output_block_offset + element_index, result)
        else:
            # Load packed element
            packed = tl.load(x_ptr + input_block_offset + element_index)

            # Unpack, dequantize, and apply scaling
            if cxpr_quantized_dtype == _QUANTIZED_DTYPE_FP4:
                x1 = _fp4_dequantize(packed >> 4) * local_absmax
                x2 = _fp4_dequantize(packed & 0x0F) * local_absmax
            else:
                x1 = _nf4_dequantize(packed >> 4) * local_absmax
                x2 = _nf4_dequantize(packed & 0x0F) * local_absmax

            # We dequantize two values with each iteration of this loop
            output_element_offset = element_index * 2

            # Store new values
            tl.store(out_ptr + output_block_offset + output_element_offset, x1)
            tl.store(out_ptr + output_block_offset + output_element_offset + 1, x2)


def dequantize_blockwise_launcher(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    code: torch.Tensor | None,
    blocksize: int,
    output_size: int,
    quant_type: str,
) -> torch.Tensor:
    """Convert FP32/FP16/BF16 input tensor to NF4/FP4."""
    # Blocksize must be a power-of-two
    assert blocksize == triton.next_power_of_2(blocksize)  # noqa: S101
    if quant_type in ("fp4", "nf4"):
        # Only support even output size if unpacking one element into two
        assert output_size % 2 == 0  # noqa: S101

    quantized_dtype = _get_quantized_dtype(quant_type)
    if quantized_dtype == QuantizedDType.GENERAL_8BIT:
        assert code is not None  # noqa: S101

    # If 4-bit, we're unpacking one element from the input into two elements of the output
    input_blocksize = blocksize if quantized_dtype == QuantizedDType.GENERAL_8BIT else blocksize // 2

    # How many blocks are we processing?
    num_blocks = triton.cdiv(output_size, blocksize)

    # Process each block separately
    grid = (num_blocks,)

    # Launch kernel
    _dequantize_blockwise_kernel[grid](
        # Input must be treated as uint8
        # See: https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#quantized-data-storage
        x_ptr=x.view(torch.uint8),
        absmax_ptr=absmax,
        out_ptr=out,
        code_ptr=code,
        output_size=output_size,
        cxpr_blocksize=blocksize,
        cxpr_input_blocksize=input_blocksize,
        cxpr_quantized_dtype=quantized_dtype.value,
    )

    return out
