# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Port of vllm gelu_tanh_and_mul to Triton."""

from typing import Final

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # type: ignore[attr-defined]

_M_2_SQRTPI: Final = 1.12837916709551257390
_M_SQRT2: Final = 1.41421356237309504880


@triton.jit  # type: ignore[misc]
def _gelu_tanh_and_mul_kernel(
    out_ptr: tl.tensor,  # [..., d]
    out_stride: int,
    in_ptr: tl.tensor,  # [..., 2 * d]
    in_stride: int,
    d: int,
    cxpr_block_size: tl.constexpr,
    cxpr_beta: tl.constexpr,
    cxpr_kappa: tl.constexpr,
) -> None:
    """Apply tanh-approximated GeLU and multiply.

    This step is part of the GeGLU activation used in the GEMMA model.
    https://arxiv.org/abs/2002.05202
    https://storage.googleapis.com/gweb-developer-goog-blog-assets/images/image2_l7UnOuC.original.png

    vllm refs:
    vllm::act_and_mul_kernel (wraps user-defined ACT and performs mul)
    vllm::gelu_tanh_kernel
    vllm::gelu_tanh_and_mul
    LAUNCH_ACTIVATION_GATE_KERNEL

    Ref implementation summary:
    - One block per token
    - Threads work in parallel to perform point-wise GeLU and multiply
    - Threads will loop until all d elements are processed

    Args:
        out_ptr: Storage for results; expected dimensions (num_tokens, projection_dim).
        out_stride: step size between output rows.
        in_ptr: Adjoined projections to be scaled [ ..., [gate_proj | up_proj]]; ([opt-batch], num_tokens, projection_dim * 2).
        in_stride: step size between input rows.
        d: dimensionality of projection space.
        cxpr_block_size: working block size.
        cxpr_beta: BETA constant.
        cxpr_kappa: KAPPA constant.
    """
    token_idx = tl.program_id(axis=0)
    in_block_start = in_ptr + (token_idx * in_stride)
    local_offset = tl.arange(0, cxpr_block_size)
    out_block_start = out_ptr + (token_idx * out_stride)
    for _ in tl.range(0, d, cxpr_block_size):
        mask = local_offset < d
        x = tl.load(in_block_start + local_offset, mask=mask)
        y = tl.load(in_block_start + d + local_offset, mask=mask)
        x_cubed = x * x * x
        inner = cxpr_beta * (x + cxpr_kappa * x_cubed)
        gelu_act = 0.5 * x * (1 + libdevice.tanh(inner))
        scaled = gelu_act * y
        tl.store(out_block_start + local_offset, scaled, mask=mask)
        local_offset += cxpr_block_size


def gelu_tanh_and_mul_launcher(
    output: torch.Tensor,
    projections: torch.Tensor,
) -> None:
    """gelu_tanh_and_mul launcher.

    Args:
        output: Storage for results; expected dimensions (num_tokens, projection_dim)
        projections: Adjoined projections to be scaled; (num_tokens, projection_dim * 2)
    """
    d = projections.shape[-1] // 2
    num_tokens = projections.numel() // (2 * d)
    beta = _M_SQRT2 * _M_2_SQRTPI * 0.5
    kappa = 0.044715
    block_size = min(256, triton.next_power_of_2(d))

    # Note about using negative indices in shape and stride
    # The commonly (?) accepted organization of data appears to be:
    # - batch, token, other higher level dimensions in the leading positions
    # - data-specific dimensions in the lower positions
    # For example: [batch_size, num_token, embedding_size]
    # Some higher level dimensions may or may not exist.  For example, for unbatched data, there wouldn't be a
    # batch_size dimension.  On the other hand, the lower-level dimensions will always exist.  This is why it makes
    # some sense to count backwards when trying to access dimension information.

    _gelu_tanh_and_mul_kernel[(num_tokens,)](
        output,
        output.stride(-2),
        projections,
        projections.stride(-2),
        d,
        cxpr_block_size=block_size,
        cxpr_beta=beta,
        cxpr_kappa=kappa,
    )
