# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of RMS norm."""

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _rms_norm_kernel(  # noqa: PLR0913
    # Tensors
    output_ptr: tl.tensor,  # [..., hidden_size]
    x_ptr: tl.tensor,  # [..., hidden_size]
    residual_ptr: tl.tensor,  # [..., hidden_size]
    weight_ptr: tl.tensor,  # [hidden_size]
    # Scalars
    hidden_size: int,
    epsilon: float,
    # Constexprs
    cxpr_block_size: tl.constexpr,
    cxpr_use_residual: tl.constexpr,
) -> None:
    """Implementation of RMS norm kernel.

    Args:
        output_ptr: Pointer to output tensor, shape: (num_tokens, hidden_size).
        x_ptr: Pointer to input tensor, shape: (num_tokens, hidden_size).
        residual_ptr: Pointer to residual tensor, shape: (num_tokens, hidden_size).
        weight_ptr: Pointer to weight tensor, shape: (hidden_size,).
        hidden_size: Hidden size.
        epsilon: Epsilon value.
        cxpr_block_size: Number of elements to process at once.
        cxpr_use_residual: Whether to use residual tensor.
    """
    token_index = tl.program_id(0)
    token_offset = token_index * hidden_size

    block_offsets = tl.arange(0, cxpr_block_size)
    mask = block_offsets < hidden_size

    x = tl.load(x_ptr + token_offset + block_offsets, mask=mask)
    w = tl.load(weight_ptr + block_offsets, mask=mask)

    if cxpr_use_residual:
        # Load residual, add it to x, and store it
        residual = tl.load(residual_ptr + token_offset + block_offsets, mask=mask)
        x += residual
        tl.store(residual_ptr + token_offset + block_offsets, x, mask=mask)

        # If we are using the residual, we will write the result to the input tensor
        output_ptr = x_ptr

    # For parity with vLLM, we will use fp32 here
    x = x.to(tl.float32)
    mean_of_squares = tl.sum(x * x) / hidden_size
    rms_inv = tl.rsqrt(mean_of_squares + epsilon)

    result = (x * rms_inv).to(x_ptr.dtype.element_ty) * w

    tl.store(output_ptr + token_offset + block_offsets, result, mask=mask)


def rms_norm_launcher(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Launch rms_norm kernel.

    Args:
        output: Output tensor, of shape (..., hidden_size).
        x: Input tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    hidden_size = x.size(-1)
    num_tokens = x.numel() // hidden_size

    assert output.shape == x.shape, "Output shape must match input shape"  # noqa: S101
    assert output.stride(-2) == x.stride(-2), "Output and input strides must match"  # noqa: S101
    assert output.stride(-2) == hidden_size, "Hidden size must match second-to-last stride of input/output"  # noqa: S101
    assert weight.size(0) == hidden_size, "Weight size must match hidden size"  # noqa: S101

    assert output.is_contiguous()  # noqa: S101
    assert x.is_contiguous()  # noqa: S101
    assert weight.is_contiguous()  # noqa: S101

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _rms_norm_kernel[grid](
        # Tensors
        output_ptr=output,
        x_ptr=x,
        residual_ptr=None,
        weight_ptr=weight,
        # Scalars
        hidden_size=hidden_size,
        epsilon=epsilon,
        # Constexprs
        # Note: we _could_ run out of shared memory here if hidden_size is too large.
        # If this is a concern, we could allocate some extra memory for the sum of squares and then
        # reduce it in a second kernel.
        cxpr_block_size=triton.next_power_of_2(hidden_size),
        cxpr_use_residual=False,
    )


def fused_add_rms_norm_launcher(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Launch rms_norm kernel.

    Args:
        x: Input tensor, of shape (..., hidden_size).
        residual: Residual tensor, of shape (..., hidden_size).
        weight: Weight tensor, of shape (hidden_size,).
        epsilon: Epsilon value.
    """
    hidden_size = x.size(-1)
    num_tokens = x.numel() // hidden_size

    assert x.shape == residual.shape, "Input shape must match residual shape"  # noqa: S101
    assert x.stride(-2) == residual.stride(-2), "Input and residual strides must match"  # noqa: S101
    assert x.stride(-2) == hidden_size, "Hidden size must match second-to-last stride of input"  # noqa: S101
    assert weight.size(0) == hidden_size, "Weight size must match hidden size"  # noqa: S101

    assert x.is_contiguous()  # noqa: S101
    assert residual.is_contiguous()  # noqa: S101
    assert weight.is_contiguous()  # noqa: S101

    # Parallelize over the number of tokens
    grid = (num_tokens,)

    # Launch kernel
    _rms_norm_kernel[grid](
        # Tensors
        output_ptr=None,
        x_ptr=x,
        residual_ptr=residual,
        weight_ptr=weight,
        # Scalars
        hidden_size=hidden_size,
        epsilon=epsilon,
        # Constexprs
        # Note: we _could_ run out of shared memory here if hidden_size is too large.
        # If this is a concern, we could allocate some extra memory for the sum of squares and then
        # reduce it in a second kernel.
        cxpr_block_size=triton.next_power_of_2(hidden_size),
        cxpr_use_residual=True,
    )
