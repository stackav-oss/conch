# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of bitsandbytes quantization."""

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
def _nf4_quantize(x: tl.tensor) -> int:  # noqa: C901, PLR0911, PLR0912
    """Quantize a floating point value to NF4."""
    if x > 0.8614784181118011:  # noqa: PLR2004
        return 15
    if x > 0.6427869200706482:  # noqa: PLR2004
        return 14
    if x > 0.5016634166240692:  # noqa: PLR2004
        return 13
    if x > 0.3893125355243683:  # noqa: PLR2004
        return 12
    if x > 0.2920137718319893:  # noqa: PLR2004
        return 11
    if x > 0.2035212516784668:  # noqa: PLR2004
        return 10
    if x > 0.1202552504837513:  # noqa: PLR2004
        return 9
    if x > 0.03979014977812767:  # noqa: PLR2004
        return 8
    if x > -0.045525018125772476:  # noqa: PLR2004
        return 7
    if x > -0.13791173323988914:  # noqa: PLR2004
        return 6
    if x > -0.23460740596055984:  # noqa: PLR2004
        return 5
    if x > -0.33967943489551544:  # noqa: PLR2004
        return 4
    if x > -0.4599952697753906:  # noqa: PLR2004
        return 3
    if x > -0.6106329262256622:  # noqa: PLR2004
        return 2
    if x > -0.8480964004993439:  # noqa: PLR2004
        return 1
    return 0


@triton.jit  # type: ignore[misc]
def _fp4_quantize(x: tl.tensor) -> int:  # noqa: C901, PLR0911, PLR0912
    """Quantize a floating point value to FP4."""
    sign = 8 if x < 0 else 0
    x = tl.abs(x)

    if x > 0.29166667:
        if x > 0.5833334:
            if x > 0.83333334:
                return 3 + sign
            return 2 + sign
        if x > 0.4166667:
            return 5 + sign
        return 4 + sign

    if x > 0.0859375:
        if x > 0.208333334:
            return 7 + sign
        return 6 + sign
    if x > 0.00260417:
        return 1 + sign
    return 0 + sign


@triton.jit  # type: ignore[misc]
def _fp8_quantize(x: tl.tensor, code_ptr: tl.tensor) -> int:
    """Quantize a floating point value to FP8 (bitcast-ed to uint8)."""
    pivot = 127
    upper_pivot = 255
    lower_pivot = 0

    lower = -1.0
    upper = 1.0

    val = tl.load(code_ptr + pivot)

    # Iterate backwards through powers of two: [64, 32, 16, 8, 4, 2, 1]
    for current_power_of_two in range(6, -1, -1):
        i = tl.exp2(current_power_of_two.to(tl.float32)).to(tl.int32)  # type: ignore[attr-defined]

        if x > val:
            lower_pivot = pivot
            lower = val
            pivot += i
        else:
            upper_pivot = pivot
            upper = val
            pivot -= i

        val = tl.load(code_ptr + pivot)

    if upper_pivot == 255:  # noqa: PLR2004
        upper = tl.load(code_ptr + upper_pivot)
    if lower_pivot == 0:
        lower = tl.load(code_ptr + lower_pivot)

    if x > val:
        midpoint = (upper + val) * 0.5
        if x > midpoint:
            return upper_pivot
        return pivot

    midpoint = (lower + val) * 0.5
    if x < midpoint:
        return lower_pivot
    return pivot


@triton.jit  # type: ignore[misc]
def _quantize_blockwise_kernel(  # noqa: PLR0913
    x_ptr: tl.tensor,
    absmax_ptr: tl.tensor,
    out_ptr: tl.tensor,
    code_ptr: tl.tensor,
    input_size: int,
    cxpr_blocksize: tl.constexpr,
    cxpr_output_blocksize: tl.constexpr,
    cxpr_quantized_dtype: tl.constexpr,
) -> None:
    """Quantize blockwise kernel."""
    block_index = tl.program_id(0)

    input_block_offset = block_index * cxpr_blocksize

    block_offsets = input_block_offset + tl.arange(0, cxpr_blocksize)
    block_mask = block_offsets < input_size

    # Load this block of the input
    block = tl.load(x_ptr + block_offsets, mask=block_mask, other=0.0)

    # Reduce absmax and store for de-quantization
    local_absmax = tl.max(tl.abs(block))
    tl.store(absmax_ptr + block_index, local_absmax)

    # Invert so that we can _multiply_ by scaling factor during quantization
    local_absmax = 1.0 / local_absmax.to(tl.float32)

    # Don't read extra elements if input_size is not perfectly divisible by blocksize
    output_block_offset = block_index * cxpr_output_blocksize
    this_block_size = min(cxpr_blocksize, input_size - input_block_offset)
    this_output_block_size = (
        this_block_size if cxpr_quantized_dtype == _QUANTIZED_DTYPE_GENERAL_8BIT else this_block_size // 2
    )

    for element_index in range(this_output_block_size):
        if cxpr_quantized_dtype == _QUANTIZED_DTYPE_GENERAL_8BIT:
            x = tl.load(x_ptr + input_block_offset + element_index)
            result = _fp8_quantize(x.to(tl.float32) * local_absmax, code_ptr)
            tl.store(out_ptr + output_block_offset + element_index, result)
        else:
            # We quantize two values with each iteration of this loop
            packed_element_offset = element_index * 2
            # Load two elements (note: unfortunately we cannot index into these elements even though we already loaded them)
            x1 = tl.load(x_ptr + input_block_offset + packed_element_offset)
            x2 = tl.load(x_ptr + input_block_offset + packed_element_offset + 1)

            # Cast to fp32, scale by inverse of max(abs()) of block
            x1 = x1.to(tl.float32) * local_absmax
            x2 = x2.to(tl.float32) * local_absmax

            # Initialize result to 0 (i.e. bitwise-or always flips the bit)
            result = 0

            # Apply quantization and bitshift
            if cxpr_quantized_dtype == _QUANTIZED_DTYPE_FP4:
                result |= _fp4_quantize(x1) << 4
                result |= _fp4_quantize(x2)
            else:
                result |= _nf4_quantize(x1) << 4
                result |= _nf4_quantize(x2)

            # Store
            tl.store(out_ptr + output_block_offset + element_index, result)


def quantize_blockwise_launcher(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    code: torch.Tensor | None,
    blocksize: int,
    input_size: int,
    quant_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert FP32/FP16/BF16 input tensor to FP4/NF4/FP8."""
    # Blocksize must be a power-of-two
    assert blocksize == triton.next_power_of_2(blocksize)  # noqa: S101
    if quant_type in ("fp4", "nf4"):
        # Only support even input size if packing two elements into one
        assert input_size % 2 == 0  # noqa: S101

    quantized_dtype = _get_quantized_dtype(quant_type)
    if quantized_dtype == QuantizedDType.GENERAL_8BIT:
        assert code is not None  # noqa: S101

    # If 4-bit, we're packing two elements from the input into each element of the output
    output_blocksize = blocksize if quantized_dtype == QuantizedDType.GENERAL_8BIT else blocksize // 2

    # How many blocks are we processing?
    num_blocks = triton.cdiv(input_size, blocksize)

    # Process each block separately
    grid = (num_blocks,)

    # Launch kernel
    _quantize_blockwise_kernel[grid](
        x_ptr=x,
        absmax_ptr=absmax,
        # Need output to be treated as uint8
        # See: https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#quantized-data-storage
        out_ptr=out.view(torch.uint8),
        code_ptr=code,
        input_size=input_size,
        cxpr_blocksize=blocksize,
        cxpr_output_blocksize=output_blocksize,
        cxpr_quantized_dtype=quantized_dtype.value,
    )

    return out, absmax
