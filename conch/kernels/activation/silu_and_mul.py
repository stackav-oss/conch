# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) and mul."""

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _silu_and_mul_kernel(  # noqa: PLR0913
    # Pointers to tensors
    output_ptr: tl.tensor,  # [..., d]
    x_ptr: tl.tensor,  # [..., 2 * d]
    # Strides of relevant tensors
    output_stride: int,
    input_stride: int,
    # Scalars
    d: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
) -> None:
    """Implementation of SiLU and multiply kernel.

    Args:
        output_ptr: Pointer to output tensor, shape: (num_tokens, d) or (batch_size, sequence_length, d).
        x_ptr: Pointer to input tensor, shape: (num_tokens, 2 * d) or (batch_size, sequence_length, 2 * d).
        output_stride: Stride of output tensor between elements in the last dimension.
        input_stride: Stride of input tensor between elements in the last dimension.
        d: Size of last dimension of output.
        cxpr_block_size: Number of elements to process at once.
    """
    token_index = tl.program_id(0)

    x_token_offset = token_index * input_stride
    y_token_offset = x_token_offset + d
    output_token_offset = token_index * output_stride

    block_offsets = tl.arange(0, cxpr_block_size)

    for _ in tl.range(0, d, cxpr_block_size):
        mask = block_offsets < d

        # For parity with vLLM, compute `x * sigmoid(x)` in fp32
        x = tl.load(x_ptr + x_token_offset + block_offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(x_ptr + y_token_offset + block_offsets, mask=mask, other=0.0)

        silu = (x * tl.sigmoid(x)).to(x_ptr.dtype.element_ty)
        silu *= y

        tl.store(output_ptr + output_token_offset + block_offsets, silu, mask=mask)

        block_offsets += cxpr_block_size


def silu_and_mul_launcher(
    output: torch.Tensor,
    x: torch.Tensor,
) -> None:
    """Launch silu_and_mul kernel.

    Args:
        x: Input tensor, of shape (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d).
        output: Output tensor, of shape (num_tokens, d) or (batch_size, seq_len, d).
    """
    d = x.size(-1) // 2
    cxpr_block_size = min(1024, triton.next_power_of_2(d))

    num_tokens = x.numel() // x.size(-1)

    assert output.shape == x.shape[:-1] + (d,), "Output shape must match input shape with last dimension halved!"  # noqa: S101
    assert x.is_contiguous()  # noqa: S101
    assert output.is_contiguous()  # noqa: S101

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _silu_and_mul_kernel[grid](
        # Tensors
        output_ptr=output,
        x_ptr=x,
        # Strides of relevant tensors
        output_stride=output.stride(-2),
        input_stride=x.stride(-2),
        # Scalars
        d=d,
        # Constexprs
        cxpr_block_size=cxpr_block_size,
    )
