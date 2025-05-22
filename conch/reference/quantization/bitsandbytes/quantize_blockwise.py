# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Bitsandbytes quantize PyTorch reference implementation."""

from collections.abc import Callable

import torch
import triton


def _pytorch_fp4_quantize(x: torch.Tensor, code: torch.Tensor | None) -> int:  # noqa: PLR0911
    """Quantize a floating point value to FP4."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is None

    sign = 8 if x < 0 else 0
    x = torch.abs(x)

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


def _pytorch_nf4_quantize(x: torch.Tensor, code: torch.Tensor | None) -> int:  # noqa: C901, PLR0911, PLR0912
    """Quantize a floating point value to NF4."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is None
    if x > 0.03979014977812767:
        if x > 0.3893125355243683:
            if x > 0.6427869200706482:
                if x > 0.8614784181118011:
                    return 15
                return 14
            if x > 0.5016634166240692:
                return 13
            return 12
        if x > 0.2035212516784668:
            if x > 0.2920137718319893:
                return 11
            return 10
        if x > 0.1202552504837513:
            return 9
        return 8
    if x > -0.33967943489551544:
        if x > -0.13791173323988914:
            if x > -0.045525018125772476:
                return 7
            return 6
        if x > -0.23460740596055984:
            return 5
        return 4
    if x > -0.6106329262256622:
        if x > -0.4599952697753906:
            return 3
        return 2
    if x > -0.8480964004993439:
        return 1
    return 0


def _pytorch_fp8_quantize(x: torch.Tensor, code: torch.Tensor | None) -> int:
    """Quantize a floating point value to FP8."""
    assert x.ndim == 0, "Argument must be scalar tensor!"
    assert code is not None
    pivot = 127
    upper_pivot = 255
    lower_pivot = 0

    lower = torch.tensor(-1.0)
    upper = torch.tensor(1.0)

    val = code[pivot]

    for i in [64, 32, 16, 8, 4, 2, 1]:
        if x > val:
            lower_pivot = pivot
            lower = val
            pivot += i
        else:
            upper_pivot = pivot
            upper = val
            pivot -= i

        val = code[pivot]

    if upper_pivot == 255:
        upper = code[upper_pivot]
    if lower_pivot == 0:
        lower = code[lower_pivot]

    if x > val:
        midpoint = (upper + val) * 0.5
        if x > midpoint:
            return upper_pivot
        return pivot

    midpoint = (lower + val) * 0.5
    if x < midpoint:
        return lower_pivot
    return pivot


def _get_pytorch_quant_method(quant_type: str) -> Callable[[torch.Tensor, torch.Tensor | None], int]:
    """Get quantization method."""
    if quant_type == "nf4":
        return _pytorch_nf4_quantize
    if quant_type == "fp4":
        return _pytorch_fp4_quantize
    return _pytorch_fp8_quantize


def quantize_blockwise_launcher(  # noqa: PLR0913
    x: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    code: torch.Tensor | None,
    blocksize: int,
    input_size: int,
    quant_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert FP32/FP16/BF16 input tensor to NF4/FP4/FP8."""
    # This function assumes we are always packing two, NF4/FP4 elements into one 8-bit integer
    assert out.dtype == torch.uint8

    quant_method = _get_pytorch_quant_method(quant_type)

    num_blocks = triton.cdiv(input_size, blocksize)
    output_blocksize = blocksize // 2 if quant_type in ("fp4", "nf4") else blocksize

    # Process in blocks of size {blocksize}
    for block_index in range(num_blocks):
        # Get elements of current block
        input_block_offset = block_index * blocksize
        # Don't read extra elements if input_size is not perfectly divisible by blocksize
        this_block_size = min(blocksize, input_size - input_block_offset)
        block = x[input_block_offset : input_block_offset + this_block_size]

        # Reduce abs(max())
        local_absmax = torch.max(torch.abs(block))

        # Record for de-quantization
        absmax[block_index] = local_absmax

        # Invert so that we can _multiply_ by scaling factor during quantization
        local_absmax = 1.0 / local_absmax.to(torch.float32)

        output_block_offset = block_index * output_blocksize

        # Don't read extra elements if input_size is not perfectly divisible by blocksize
        this_output_block_size = this_block_size // 2 if quant_type in ("fp4", "nf4") else this_block_size

        # Pack two quantized elements into one for the output (two 4-bit values packed into a `uint8`)
        for element_index in range(this_output_block_size):
            if quant_type in ("fp4", "nf4"):
                # Initialize result to 0 (i.e. bitwise-or always flips the bit)
                result = 0
                # Pack first element (note: for parity with BNB, upcast to fp32)
                result |= quant_method(block[element_index * 2].to(torch.float32) * local_absmax, code) << 4
                # Pack second element (same upcasting as above)
                result |= quant_method(block[element_index * 2 + 1].to(torch.float32) * local_absmax, code)
            else:
                result = quant_method(block[element_index].to(torch.float32) * local_absmax, code)

            # Store result into output
            out[output_block_offset + element_index] = result

    return out, absmax
