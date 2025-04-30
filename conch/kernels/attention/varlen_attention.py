# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton implementation of variable-length Flash Attention."""

from typing import Final

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # type: ignore[attr-defined]

from conch.platforms import current_platform

"""

Potential strategy:
    - Use same idea of PagedAttention kernel to split into two stages
    - Split query into separate sequences
    - Then split each sequence into {seqlen_q_block_size} components


Q = [
  # Seq_0
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  # Seq_1
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  # Seq_2
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
]

Seq_0: 6 tokens
Seq_1: 2 tokens
Seq_2: 8 tokens

v1:
    - Split each sequence
    - For short sequences: split query vectors into groups of size==16 (for MMA)
    -

v2:
    - For longer prefills (seqlen > X?) then we split KV into chunks


Llama-3.1-8B:
    num_query_heads = 32
    num_kv_heads = 8

    query_group_size == 4
        -> Every KV head would be loaded 4 times if we just group queries

Llama-3.1-405B:
    num_query_heads: 128
    num_kv_heads: 8

    query_group_size == 16
        -> Every KV head would be loaded 16 times if we just group queries

    If query_group_size >= seqlen_q_block_size then it'd probably better to do the query group together?


If group_size = 16 and seqlen_q = 8:
    q_block = [seq0_head0, seq1_head0, seq2_head0, ... seq7_head0, X, X, X... X]

    -> 16 iterations (one for each head), 16 results per head (but 8 are masked out)

OR
    q_block = [seq0_head0, seq0_head1, seq0_head2, ... seq0_head15]

    -> 8 iterations (one for each token), 16 results per token (no masking)






IDEA:
    - What if instead of loading Q and then reloading KV repeatedly, what if we loaded KV and then iterated through Q?

    so:

K = [
  # Seq_0
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  # ...
  # N-times
  #
  # Seq_1
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  # ...
  # M-times
  #
  # Seq_2
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  [[...,  head_size], ..., num_query_heads],
  # K-times
]


Then we could load a chunk of K/V and process all of the different queries for that block?



QUESTION: WHEN IS SEQLEN_Q != SEQLEN_K?????



For very short sequences:
    - It's likely more efficient to load groups for MQA/GQA
    - Each KV head processes [1-{query_group_size}] query heads

For MHA:
    - num_query_heads == num_kv_heads, so it doesn't even matter

Option:
    - Have separate kernels for MHA and GQA (probably better to just call it GQA bc MQA is a special case of GQA where query_group_size == num_query_heads?)

"""


@triton.jit  # type: ignore[misc]
def _varlen_attention_kernel(  # noqa: PLR0913, PLR0915
    # Pointers to tensors
    output_ptr: tl.tensor,
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    cu_seqlen_q_ptr: tl.tensor,
    cu_seqlen_k_ptr: tl.tensor,
    # Scalar arguments
    query_group_size: int,
    head_size: int,
    # max_seqlen: int,
    # Stride arguments
    query_sequence_stride: int,
    query_head_stride: int,
    kv_sequence_stride: int,
    kv_head_stride: int,
    # Constexpr arguments
    # cxpr_is_varlen: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
    cxpr_seqlen_q_block_size: tl.constexpr,
    cxpr_seqlen_kv_block_size: tl.constexpr,
    # cxpr_query_group_size_padded: tl.constexpr,
) -> None:
    """varlen attention kernel."""
    # What sequence is this program processing?
    sequence_index = tl.program_id(0)
    # What block of the sequence is this program processing?
    sequence_block_index = tl.program_id(1)
    # What query head is this program processing?
    query_head_index = tl.program_id(2)

    # How many blocks are we splitting each sequence into?
    num_blocks_per_sequence = tl.num_programs(1)

    dtype = output_ptr.dtype.element_ty

    # What KV head is this program processing?
    kv_head_index = query_head_index // query_group_size

    seq_q_start = tl.load(cu_seqlen_q_ptr + sequence_index)
    seq_q_end = tl.load(cu_seqlen_q_ptr + sequence_index + 1)
    this_seqlen_q = seq_q_end - seq_q_start

    seq_kv_start = tl.load(cu_seqlen_k_ptr + sequence_index)
    seq_kv_end = tl.load(cu_seqlen_k_ptr + sequence_index + 1)
    this_seqlen_kv = seq_kv_end - seq_kv_start

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
    output = tl.zeros([cxpr_seqlen_q_block_size, cxpr_head_size_padded], dtype=dtype)
    # Keep running max of softmax numerator (scale * Q * K)
    m_i = tl.full([cxpr_seqlen_q_block_size], -float("inf"), dtype=dtype)
    # Keep running denominator of softmax
    l_i = tl.full([cxpr_seqlen_q_block_size], 0.0, dtype=dtype)

    query_head_index_offset = query_head_index * query_head_stride

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    # TODO(jmanning): Handle if not full
    # this_seq_group_size = cxpr_seq_block_size
    this_sequence_block_offset = sequence_block_index * cxpr_seqlen_q_block_size
    if this_sequence_block_offset > this_seqlen_q:
        return

    # Calculate number of entries in this cache block (will be a value between 1 and cache_block_size)
    this_sequence_block_size = min(this_seqlen_q - this_sequence_block_offset, cxpr_seqlen_q_block_size)

    # Offsets for each query vector in the sequence block
    sequence_block_offsets = tl.arange(0, cxpr_seqlen_q_block_size)
    # Mask out any items in the block that aren't used
    sequence_block_mask = sequence_block_offsets < this_sequence_block_size

    # Offsets for the queries in this block
    query_offsets = (
        (sequence_block_offsets[:, None] * query_sequence_stride)
        + (query_head_index * query_head_stride)
        + head_offsets[None, :]
    )
    # Mask out unused query vectors and unused head elements
    query_mask = sequence_block_mask[:, None] & head_mask[None, :]

    # Offsets for query vector this batch/head
    # query_batch_index_offset = batch_index * query_batch_stride
    # query_head_index_offset = query_head_index * query_head_stride

    query_sequence_offset = seq_q_start * query_sequence_stride
    # query_head_offset = query_head_index * query_head_stride

    # if sequence_index == 0:
    #     if sequence_block_index == 0:
    #         if query_head_index == 0:
    #             print("query_offsets = ", query_offsets)

    query = tl.load(query_ptr + query_sequence_offset + query_offsets, mask=query_mask, other=0.0)

    # Load the key block as (cxpr_seqlen_q_block_size, cxpr_head_size_padded)
    # query_block_ptr = tl.make_block_ptr(
    #     # key_cache_ptr + kv_cache_block_index_offset + kv_head_index_offset,
    #     query_ptr + query_sequence_offset,
    #     shape=(this_sequence_block_size, head_size),
    #     # strides=(kv_head_element_stride, kv_cache_block_stride),
    #     # strides=(query_head_stride, query_element_stride),
    #     strides=(query_sequence_stride, query_element_stride),
    #     offsets=(query_head_index * query_head_stride, 0),
    #     block_shape=(cxpr_seqlen_q_block_size, cxpr_head_size_padded),
    #     order=(0, 1),
    # )
    # query_block = tl.load(query_block_ptr, boundary_check=(0, 1), padding_option="zero")

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
    assert output.dtype == query.dtype  # noqa: S101

    assert query.stride(0) == output.stride(0)  # noqa: S101
    assert query.stride(1) == output.stride(1)  # noqa: S101
    assert query.stride(2) == output.stride(2)  # noqa: S101
    assert key.stride(0) == value.stride(0)  # noqa: S101
    assert key.stride(1) == value.stride(1)  # noqa: S101
    assert key.stride(2) == value.stride(2)  # noqa: S101

    # assert cu_seqlen_q is not None, "cu_seqlen_q must be provided if cu_seqlen_k is provided."
    # assert cu_seqlen_k is not None, "cu_seqlen_k must be provided if cu_seqlen_q is provided."
    assert len(cu_seqlen_q) == len(cu_seqlen_k), "cu_seqlen_q and cu_seqlen_k must have the same length."
    # assert cu_seqlen_q == cu_seqlen_k, "cu_seqlen_q and cu_seqlen_k must be equal."

    assert max_seqlen_q > 0, "max_seqlen_q must be positive."
    assert max_seqlen_k > 0, "max_seqlen_k must be positive."
    # assert max_seqlen_q == max_seqlen_k, "max_seqlen_q and max_seqlen_k must be equal."

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
    query_group_size = num_query_heads // num_kv_heads
    # We pad this size to be at least 16 so that we can use `tl.dot()` operations inside of the kernel
    # cxpr_query_group_size_padded = max(16, triton.next_power_of_2(query_group_size))

    # TODO(jmanning): Tune these
    cxpr_seqlen_q_block_size = 16
    cxpr_seqlen_kv_block_size = 16

    num_seqlen_q_blocks = triton.cdiv(max_seqlen_q, cxpr_seqlen_q_block_size)
    # num_seqlen_kv_blocks = triton.cdiv(max_seqlen_k, cxpr_seqlen_kv_block_size)

    # grid = (batch_size, num_seqlen_q_blocks * num_seqlen_kv_blocks, num_query_heads)
    grid = (batch_size, num_seqlen_q_blocks, num_query_heads)

    # Launch kernel
    _varlen_attention_kernel[grid](
        output_ptr=output,
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        cu_seqlen_q_ptr=cu_seqlen_q,
        cu_seqlen_k_ptr=cu_seqlen_k,
        # num_query_heads=num_query_heads,
        # num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        head_size=head_size,
        # max_seqlen=max_seqlen_q,
        # max_seqlen=max_seqlen_q,
        query_sequence_stride=query.stride(0),
        query_head_stride=query.stride(1),
        kv_sequence_stride=key.stride(0),
        kv_head_stride=key.stride(1),
        cxpr_head_size_padded=cxpr_head_size_padded,
        cxpr_seqlen_q_block_size=cxpr_seqlen_q_block_size,
        cxpr_seqlen_kv_block_size=cxpr_seqlen_kv_block_size,
        # cxpr_query_group_size_padded=cxpr_query_group_size_padded,
    )
