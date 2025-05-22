# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Port of vllm gemma rms norm to Triton."""

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _gemma_rms_norm_inplace_kernel(
    x_ptr: tl.tensor, weights_ptr: tl.const, hidden_size: int, eps: float, cxpr_block_size: tl.constexpr
) -> None:
    """Perform Gemma's version of RMS layer norm.

    Args:
        x_ptr: Tensor to be normalized, shape: (num_tokens, hidden_size).
        weights_ptr: learned weights of norm layer, shape: (hidden_size,).
        hidden_size: hidden size, often corresponds to head size.
        eps: value to pad during inverse-rms calculation.
        cxpr_block_size: must be next-power-of-two >= hidden_size.
    """
    # One block per row/token
    token_id = tl.program_id(0)
    token_row_ptr = x_ptr + token_id * hidden_size

    # Gemma RMS details
    # - Compute in float32, then convert back to original type
    # - mean-of-squares = mean(x ** 2)
    # - root-of-mean-of-squares (RMS) = sqrt(mean-of-squares + eps)
    # - weigh scaled inputs: x = (x / RMS) * (1 + weight)
    # fp32 as working format is discussed here:
    # https://github.com/huggingface/transformers/pull/29402
    # It would be reasonable do something a little bit more precise,
    # i.e. basing the precision on the input precision.  On the
    # other hand, fp32 is probably the most efficient type on most
    # GPUs' compute cores.

    offsets = tl.arange(0, cxpr_block_size)
    mask = offsets < hidden_size
    x = tl.load(token_row_ptr + offsets, mask=mask).to(tl.float32)
    x_sq = x * x
    mean_squares = tl.sum(x_sq) / hidden_size
    recip_rms = tl.rsqrt(mean_squares + eps)
    w = tl.load(weights_ptr + offsets, mask=mask).to(tl.float32)
    x = x * recip_rms * (1.0 + w)
    tl.store(token_row_ptr + offsets, x, mask=mask)  # Implicit cast back to original dtype


def gemma_rms_norm_inplace_launcher(
    x: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Perform Gemma's version of RMS layer norm.

    Args:
        x: Tensor to be normalized, shape: (num_tokens, hidden_size).
        weights: learned weights of norm layer, shape: (hidden_size,).
        epsilon: value to pad during inverse-rms calculation.
    """
    # Sanity check that the hidden dimensions match
    hidden_size = x.shape[-1]
    if hidden_size != weights.shape[-1]:
        msg = f"Input hidden dimenson ({hidden_size}) does not match length of weights ({weights.shape[-1]})"
        raise ValueError(msg)

    # Only support two-dimensional x
    if len(x.shape) != 2:  # noqa: PLR2004
        msg = f"x is {len(x.shape)}-dimensional.  Only supporting two dimensions."
        raise ValueError(msg)

    block_size = triton.next_power_of_2(hidden_size)
    grid = (x.shape[0],)
    _gemma_rms_norm_inplace_kernel[grid](
        x_ptr=x,
        weights_ptr=weights,
        hidden_size=hidden_size,
        eps=epsilon,
        cxpr_block_size=block_size,
    )
