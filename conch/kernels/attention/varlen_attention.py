# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton implementation of variable-length Flash Attention."""

from typing import Final

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # type: ignore[attr-defined]

from conch.platforms import current_platform


@triton.jit  # type: ignore[misc]
def _varlen_attention_kernel(  # noqa: PLR0913, PLR0915
    # Pointers to tensors
    output_ptr: tl.tensor,
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    cu_seqlen_ptr: tl.tensor,
    # Scalar arguments
    query_group_size: int,
    head_size: int,
    max_seqlen: int,
    # Stride arguments
    query_batch_stride: int,
    query_head_stride: int,
    # Constexpr arguments
    cxpr_is_varlen: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
    cxpr_seq_block_size: tl.constexpr,
    # cxpr_query_group_size_padded: tl.constexpr,
) -> None:
    """varlen attention kernel."""
    # What batch is this program processing?
    batch_index = tl.program_id(0)
    # What block of the sequence is this program processing?
    seq_block_index = tl.program_id(1)
    # What query head is this program processing?
    query_head_index = tl.program_id(2)

    # What KV head is this program processing?
    kv_head_index = query_head_index // query_group_size

    assert cxpr_is_varlen

    seq_start = tl.load(cu_seqlen_ptr + batch_index)
    seq_end = tl.load(cu_seqlen_ptr + batch_index + 1)
    seqlen = seq_end - seq_start

    # if cxpr_is_varlen:
    #     seq_start = tl.load(cu_seqlen_ptr + batch_index)
    #     seq_end = tl.load(cu_seqlen_ptr + batch_index + 1)
    #     seqlen = seq_end - seq_start
    # else:
    #     seqlen = max_seqlen

    # TODO(jmanning): How do we avoid duplicate loads of K/V for the same query head?

    # TODO(jmanning): For multi-head cases we could remove the extra axis here. But if we can make this work,
    # then for GQA/MQA cases we'd avoid duplicate loads

    # TODO(jmanning): 3D is batched matrix multiplication.
    # -> is that what we want?
    # [16, 1, 128] @ [128, 1, 16] -> [16, 1, 16]
    # [16, 1, 128] @ [16, 128, 1] -> [16, 1, 1]
    # Output from this set of tokens
    # output = tl.zeros([cxpr_seq_block_size, cxpr_query_group_size_padded, cxpr_head_size_padded], dtype=dtype)
    output = tl.zeros([cxpr_seq_block_size, cxpr_head_size_padded], dtype=dtype)
    # Keep running max of softmax numerator (scale * Q * K)
    m_i = tl.full([cxpr_seq_block_size], -float("inf"), dtype=dtype)
    # Keep running denominator of softmax
    l_i = tl.full([cxpr_seq_block_size], 0.0, dtype=dtype)

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    # TODO(jmanning): Handle if not full
    this_seq_group_size = cxpr_seq_block_size

    # Offsets for each query vector in the group
    seq_block_offsets = tl.arange(0, cxpr_seq_block_size)
    # Mask out query heads that are just for padding
    seq_block_mask = seq_block_offsets < this_seq_group_size

    # Offsets for the queries in this block
    query_offsets = seq_block_offsets[:, None] * query_head_stride + head_offsets[None, :]
    # Mask out
    query_mask = seq_block_mask[:, None] & head_mask[None, :]

    # Offsets for query vector this batch/head
    query_batch_index_offset = batch_index * query_batch_stride
    query_head_index_offset = query_head_index * query_head_stride

    # Load queries for all of the query heads that correspond to this KV head
    query = tl.load(
        query_ptr + query_batch_index_offset + query_head_index_offset + query_offsets, mask=query_mask, other=0.0
    )

    # Iterate through the sequence in blocks

    # Store output
    # Calculate offset for the sequence block
    # Calculate offset for the query head
    # Calculate offsets for each element of the head


def varlen_attention_launcher(
    output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlen_q: torch.Tensor,
    cu_seqlen_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    softcap: float = 0.0,
) -> None:
    """Launcher for varlen attention kernel.

    batch_size=2

    Q = [
      [[1, 2, ... head_size], ..., num_query_heads]],
      [[3, 4, ... head_size], ..., num_query_heads]],
    ]



    Args:
        output: Output tensor, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key: Key tensor, shape: (total_num_k, num_kv_heads, head_size).
        value: Value tensor, shape: (total_num_k, num_kv_heads, head_size).
    """
    assert output.shape == query.shape, "Output and query tensors must have the same shape."
    assert key.shape == value.shape, "Key and value tensors must have the same shape."
    assert softcap >= 0.0, "Softcap must be non-negative."

    allowed_in_out_dtypes: Final = [torch.float32, torch.float16, torch.bfloat16]
    assert query.dtype in allowed_in_out_dtypes  # noqa: S101
    assert out.dtype == query.dtype  # noqa: S101

    assert cu_seqlen_q is not None, "cu_seqlen_q must be provided if cu_seqlen_k is provided."
    assert cu_seqlen_k is not None, "cu_seqlen_k must be provided if cu_seqlen_q is provided."
    assert len(cu_seqlen_q) == len(cu_seqlen_k), "cu_seqlen_q and cu_seqlen_k must have the same length."
    assert cu_seqlen_q == cu_seqlen_k, "cu_seqlen_q and cu_seqlen_k must be equal."

    assert max_seqlen_q > 0, "max_seqlen_q must be positive."
    assert max_seqlen_k > 0, "max_seqlen_k must be positive."
    assert max_seqlen_q == max_seqlen_k, "max_seqlen_q and max_seqlen_k must be equal."

    if softcap > 0.0:
        error_msg = "Softcap is not supported yet."
        raise NotImplementedError(error_msg)

    # For prefill, query_seq_len == kv_seq_len, though this may not always be true for other attention?
    

    batch_size = len(cu_seqlen_q) - 1
    _, num_query_heads, head_size = output.shape
    _, num_kv_heads, _ = key.shape

    # Pad the head size to be a power of 2
    cxpr_head_size_padded = triton.next_power_of_2(head_size)

    # How many query heads correspond to the same KV head?
    # query_group_size = num_query_heads // num_kv_heads
    # We pad this size to be at least 16 so that we can use `tl.dot()` operations inside of the kernel
    # cxpr_query_group_size_padded = max(16, triton.next_power_of_2(query_group_size))

    # TODO(jmanning): Tune this
    cxpr_seq_block_size = 16
    num_sequence_blocks = triton.cdiv(max_seqlen_q, cxpr_seq_block_size)

    grid = (batch_size, num_sequence_blocks, num_query_heads)

    # Launch kernel
    _varlen_attention_kernel[grid](
        output_ptr=output,
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        # num_query_heads=num_query_heads,
        # num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        head_size=head_size,
        max_seqlen=max_seqlen_q,
        query_batch_stride=query.stride(0),
        query_head_stride=query.stride(1),
        cxpr_head_size_padded=cxpr_head_size_padded,
        cxpr_seq_block_size=cxpr_seq_block_size,
        # cxpr_query_group_size_padded=cxpr_query_group_size_padded,
    )
