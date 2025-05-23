# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of FP8 CompressedTensors scaling kernels."""

import torch
import triton
import triton.language as tl

from conch.platforms import current_platform


@triton.jit  # type: ignore[misc]
def _static_scaled_fp8_quant_kernel(
    # Pointers to tensors
    output_ptr: tl.tensor,  # (num_tokens, hidden_size)
    input_ptr: tl.tensor,  # (num_tokens, hidden_size)
    scale_ptr: tl.tensor,  # (1,)
    # Scalar arguments
    hidden_size: int,
    # Constexprs
    cxpr_hidden_size_padded: tl.constexpr,
    cxpr_block_size: tl.constexpr,
    cxpr_is_rocm: tl.constexpr,
) -> None:
    """FP8 quantization kernel using static scaling.

    Args:
        output_ptr: Pointer to tensor for output, shape: (num_tokens, hidden_size).
        input_ptr: Pointer to tensor for fp input, shape: (num_tokens, hidden_size).
        scale_ptr: Pointer to static scale factor, shape: (1,).
        hidden_size: Second dimension of input/output tensors.
        cxpr_hidden_size_padded: Hidden size padded to next power-of-two.
        cxpr_block_size: Block size to iterate through the hidden size for each token.
        cxpr_is_rocm: Whether or not we're on AMD.
    """
    # Program id determines which token we are processing
    token_idx = tl.program_id(0)
    # Calculate offset to this token from the start of the input/output tensors
    token_offset = token_idx * hidden_size

    # Common offsets that can be shared for each block
    block_offsets = tl.arange(0, cxpr_block_size)

    fp8_dtype = tl.float8e4b8 if cxpr_is_rocm else tl.float8e4nv

    # Invert scale factor so we can multiply instead of divide
    inverted_scale = 1.0 / tl.load(scale_ptr)

    # Iterate through the hidden_size for this token in chunks of size cxpr_block_size
    for hidden_start_idx in tl.static_range(0, cxpr_hidden_size_padded, cxpr_block_size):
        # Calculate absolute offsets and mask for this block
        hidden_offsets = hidden_start_idx + block_offsets
        mask = hidden_offsets < hidden_size
        offsets = token_offset + hidden_offsets

        # Load block from input tensor
        block = tl.load(input_ptr + offsets, mask=mask)
        # Apply inverted scaling factor and cast to FP8
        block = (block * inverted_scale).to(fp8_dtype)
        # Store result to output tensor
        tl.store(output_ptr + offsets, block, mask=mask)


def static_scaled_fp8_quant_launcher(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Launch Triton kernel to perform static-scaled fp8 quantization.

    Args:
        output_tensor: Tensor to write the output of the scaling, shape: (num_tokens, hidden_size).
        input_tensor: Tensor with input to scale, shape: (num_tokens, hidden_size).
        scale: Tensor with static scaling factor to apply, shape: (1).
    """
    num_tokens, hidden_size = output_tensor.shape

    # Triton requires power-of-two sizes
    hidden_size_padded = triton.next_power_of_2(hidden_size)
    block_size: tl.constexpr = min(hidden_size_padded, 1024)

    is_rocm: tl.constexpr = current_platform.is_amd()

    # Parallelize over the number of tokens in the sequence
    grid = (num_tokens,)

    # Launch kernel
    _static_scaled_fp8_quant_kernel[grid](
        output_ptr=output_tensor,
        input_ptr=input_tensor,
        scale_ptr=scale,
        hidden_size=hidden_size,
        cxpr_hidden_size_padded=hidden_size_padded,
        cxpr_block_size=block_size,
        cxpr_is_rocm=is_rocm,
    )
