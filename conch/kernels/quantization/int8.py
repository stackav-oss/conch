# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton implementation of Int8 CompressedTensors scaling kernels."""

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _static_scaled_int8_quant_kernel(
    # Pointers to tensors
    output_ptr: tl.tensor,  # (num_tokens, hidden_size)
    input_ptr: tl.tensor,  # (num_tokens, hidden_size)
    # Scalar arguments
    inverted_scale: float,
    hidden_size: int,
    int8_min: int,
    int8_max: int,
    # Constexprs
    cxpr_hidden_size_padded: tl.constexpr,
    cxpr_block_size: tl.constexpr,
) -> None:
    """Int8 quantization kernel using static scaling.

    Args:
        output_ptr: Pointer to tensor for output, shape: (num_tokens, hidden_size).
        input_ptr: Pointer to tensor for fp input, shape: (num_tokens, hidden_size).
        inverted_scale: Static scale factor (inverted, to avoid division).
        hidden_size: Second dimension of input/output tensors.
        int8_min: Minimum value of int8 for clamping.
        int8_max: Maximum value of int8 for clamping.
        cxpr_hidden_size_padded: Hidden size padded to next power-of-two.
        cxpr_block_size: Block size to iterate through the hidden size for each token.
    """
    # Program id determines which token we are processing
    token_idx = tl.program_id(0)
    # Calculate offset to this token from the start of the input/output tensors
    token_offset = token_idx * hidden_size

    # Common offsets that can be shared for each block
    block_offsets = tl.arange(0, cxpr_block_size)

    # Iterate through the hidden_size for this token in chunks of size cxpr_block_size
    for hidden_start_idx in tl.static_range(0, cxpr_hidden_size_padded, cxpr_block_size):
        # Calculate absolute offsets and mask for this block
        hidden_offsets = hidden_start_idx + block_offsets
        mask = hidden_offsets < hidden_size
        offsets = token_offset + hidden_offsets

        # Load block from input tensor
        block = tl.load(input_ptr + offsets, mask=mask)
        # Apply scaling factor, round, clamp, and cast to int8
        block = tl.clamp(block * inverted_scale, int8_min, int8_max).to(tl.int8)
        # Store result to output tensor
        tl.store(output_ptr + offsets, block, mask=mask)


def static_scaled_int8_quant_launcher(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Launch Triton kernel to perform static-scaled int8 quantization.

    Args:
        output_tensor: Tensor to write the output of the scaling, shape: (num_tokens, hidden_size).
        input_tensor: Tensor with input to scale, shape: (num_tokens, hidden_size).
        scale: Tensor with static scaling factor to apply, shape: (1).
    """
    num_tokens, hidden_size = output_tensor.shape

    # Need to know min/max value for int8 for clamping inside of kernel
    int8_traits = torch.iinfo(torch.int8)

    # Triton requires power-of-two sizes
    hidden_size_padded = triton.next_power_of_2(hidden_size)
    block_size: tl.constexpr = min(hidden_size_padded, 1024)

    # Parallelize over the number of tokens in the sequence
    grid = (num_tokens,)

    # Launch kernel
    _static_scaled_int8_quant_kernel[grid](
        output_ptr=output_tensor,
        input_ptr=input_tensor,
        inverted_scale=1.0 / scale.item(),
        hidden_size=hidden_size,
        int8_min=int8_traits.min,
        int8_max=int8_traits.max,
        cxpr_hidden_size_padded=hidden_size_padded,
        cxpr_block_size=block_size,
    )
