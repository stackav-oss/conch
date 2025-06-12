# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of Flash Attention w/ Paged KV Cache + FlashDecoding.

Compatible with A10, H100, AMD MI300X.
"""

from typing import Final

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice  # type: ignore[attr-defined]

# FP8 representation on CUDA and ROCm
_FP8_DTYPES: Final = [torch.float8_e4m3fn, torch.float8_e4m3fnuz]


@triton.jit  # type: ignore[misc]
def _varlen_attention_compute_splits_kernel(  # noqa: PLR0913, PLR0915
    # Pointers to tensors
    output_scratchpad_ptr: tl.tensor,  # (num_kv_splits, total_num_q, num_query_heads, head_size)
    lse_scratchpad_ptr: tl.tensor,  # (num_kv_splits, total_num_q, num_query_heads)
    query_ptr: tl.tensor,  # (total_num_q, num_query_heads, head_size)
    key_cache_ptr: tl.tensor,  # (num_cache_blocks, cache_block_size, num_kv_heads, head_size)
    value_cache_ptr: tl.tensor,  # (num_cache_blocks, cache_block_size, num_kv_heads, head_size)
    block_table_ptr: tl.tensor,  # (batch_size, max_num_blocks_per_sequence)
    seq_lens_ptr: tl.tensor,  # (batch_size, )
    cu_seqlens_q_ptr: tl.tensor,  # (batch_size + 1, )
    q_scale_ptr: tl.tensor,  # (1,)
    k_scale_ptr: tl.tensor,  # (1,)
    v_scale_ptr: tl.tensor,  # (1,)
    # Scalar arguments
    scale: float,
    num_cache_blocks_per_split: int,
    softcap: float,
    # Sizes of tensors above
    head_size: int,  # output.shape[3]
    query_group_size: int,  # num_query_heads // num_kv_heads
    num_kv_splits: int,
    # Strides for tensors above
    output_scratchpad_kv_split_stride: int,  # output_scratchpad.stride(0)
    output_scratchpad_batch_stride: int,  # output_scratchpad.stride(1)
    output_scratchpad_head_stride: int,  # output_scratchpad.stride(2)
    lse_scratchpad_kv_split_stride: int,  # lse_scratchpad.stride(0)
    lse_scratchpad_batch_stride: int,  # lse_scratchpad.stride(1)
    query_batch_stride: int,  # query.stride(0)
    query_head_stride: int,  # query.stride(1)
    kv_page_stride: int,  # key_cache.stride(0), same for key and value
    kv_cache_block_stride: int,  # key_cache.stride(1), same for key and value
    kv_head_stride: int,  # key_cache.stride(2), same for key and value
    kv_head_element_stride: int,  # key_cache.stride(3), same for key and value
    block_table_batch_stride: int,  # block_table.stride(0)
    # Constexprs
    cxpr_query_group_size_padded: tl.constexpr,  # num_query_heads // num_kv_heads
    cxpr_query_chunk_size: tl.constexpr,
    cxpr_cache_block_size: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
    cxpr_is_softcap: tl.constexpr,
    cxpr_apply_fp8_scaling: tl.constexpr,
    cxpr_is_causal: tl.constexpr,
    cxpr_split_kv: tl.constexpr,
) -> None:
    """Varlen Attention kernel: compute attention for a split block.

    Args:
        output_scratchpad_ptr: Pointer to tensor as scratchpad for output of each cache block, shape: (num_kv_splits, total_num_q, num_query_heads, head_size).
        lse_scratchpad_ptr: Pointer to tensor as scratchpad for log-sum-exp of each cache block, shape: (num_kv_splits, total_num_q, num_query_heads).
        query_ptr: Pointer to tensor storing the query, shape: (total_num_q, num_query_heads, head_size).
        key_cache_ptr: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache_ptr: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table_ptr: Pointer to tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens_ptr: Pointer to tensor holding the current sequence length for each sequence in the batch, shape: (batch_size, ).
        cu_seqlens_q_ptr: Pointer to tensor holding the cumulative sequence lengths for each sequence in the batch, shape: (batch_size, ).
        q_scale_ptr: Pointer to scalar fp8 scaling factor for q.
        k_scale_ptr: Pointer to scalar fp8 scaling factor for k.
        v_scale_ptr: Pointer to scalar fp8 scaling factor for v.
        scale: Scaling factor, 1/sqrt(head_size).
        num_cache_blocks_per_split: The maximum number of cache blocks in each split (max num each kernel will process).
        softcap: Logits softcap to apply.
        head_size: Actual head dim, not padded to power-of-two.
        query_group_size: Number of query heads in each group.
        num_kv_splits: The number of times we're splitting processing of the KV seqlen.
        output_scratchpad_kv_split_stride: Stride of the output scratchpad tensor in the 0th dimension.
        output_scratchpad_batch_stride: Stride of the output scratchpad tensor in the 1st dimension.
        output_scratchpad_head_stride: Stride of the output scratchpad tensor in the 2nd dimension.
        lse_scratchpad_kv_split_stride: Stride of the log-sum-exp scratchpad tensor in the 0th dimension.
        lse_scratchpad_batch_stride: Stride of the log-sum-exp scratchpad tensor in the 1st dimension.
        query_batch_stride: Stride of the query tensor in the 0th dimension.
        query_head_stride: Stride of the query tensor in the 1st dimension.
        kv_page_stride: Stride of the k/v tensors in the 0th dimension.
        kv_cache_block_stride: Stride of the k/v tensors in the 1st dimension.
        kv_head_stride: Stride of the k/v tensors in the 2nd dimension.
        kv_head_element_stride: Stride of the k/v tensors in the 3rd dimension.
        block_table_batch_stride: Stride of the block table tensor in the 0th dimension.
        cxpr_query_group_size_padded: The number of query heads to group together (must be power-of-two!).
        cxpr_query_chunk_size: The size of the query chunks (must be power of two!).
        cxpr_cache_block_size: The size of the cache blocks (must be power of two!).
        cxpr_head_size_padded: The head size of the attention layer padded to the next power of two.
        cxpr_is_softcap: Whether or not logits softcapping will be applied.
        cxpr_apply_fp8_scaling: Whether or not to apply FP8 scaling.
        cxpr_is_causal: Whether or not to apply causal masking.
    """
    # Encode number of query and KV splits in axis=0
    split_index = tl.program_id(0)
    total_num_splits = tl.num_programs(0)

    # kv_split_index: What "KV split" of the overall data (between 1 and N KV cache blocks) is this program processing?
    kv_split_index = split_index // tl.cdiv(total_num_splits, num_kv_splits)
    # query_split_index: What "query split" of the overall data (between 1 and M chunks of the query sequence) is this program processing?
    query_split_index = split_index % tl.cdiv(total_num_splits, num_kv_splits)

    # What KV head is this program processing?
    kv_head_index = tl.program_id(1)

    # What batch does the query that this program is processing belong to?
    batch_index = tl.program_id(2)

    # Get type that we should be using for accumulating results/intermediate calculations
    dtype = output_scratchpad_ptr.dtype.element_ty

    # Load scalar current_sequence_length for the current sequence in the batch
    # This is the KV sequence length, not the Q sequence length
    current_sequence_length = tl.load(seq_lens_ptr + batch_index)

    # The length of the current sequence will tell us how many cache blocks we need to read
    current_seq_num_cache_blocks = tl.cdiv(current_sequence_length, cxpr_cache_block_size)

    # What is the first cache block index in the set for this split
    starting_cache_block_index = kv_split_index * num_cache_blocks_per_split

    # We launch the same number of splits for all sequences in the batch, so different kernel launches will have different numbers of cache blocks
    # to process. So we may have a case where one sequence in a batch has sufficiently fewer cache blocks to process such that a kernel doesn't have
    # any blocks to process.
    if starting_cache_block_index >= current_seq_num_cache_blocks:
        return

    # Compute length of current sequence's query
    this_query_start = tl.load(cu_seqlens_q_ptr + batch_index)
    this_query_end = tl.load(cu_seqlens_q_ptr + batch_index + 1)
    this_query_length = this_query_end - this_query_start

    # If the query length is just one token, then we have a decode case
    is_pure_decode = this_query_length == 1

    # Offset for how many tokens in query correspond to previous splits for this sequence
    this_query_split_offset = query_split_index * cxpr_query_chunk_size

    # Similar to above, we launch the same number of splits for all sequences in the batch, so different kernel launches will have different
    # numbers of query tokens to process. If we've already processed all of the query tokens for this sequence, we can skip this kernel.
    if this_query_split_offset > this_query_length:
        return

    # What is the last Q token in this block?
    end_seqlen_q = this_query_split_offset + cxpr_query_chunk_size

    # How many tokens of K/V have we already processed prior to this kernel
    beginning_seqlen_k = starting_cache_block_index * cxpr_cache_block_size

    num_cache_blocks_to_process = num_cache_blocks_per_split

    if cxpr_is_causal and not is_pure_decode:
        if end_seqlen_q <= beginning_seqlen_k:
            return

        # How many cache blocks do we need to process until seqlen_k > end_seqlen_q
        num_cache_blocks_to_process = min(
            num_cache_blocks_per_split, tl.cdiv(end_seqlen_q - beginning_seqlen_k, cxpr_cache_block_size)
        )
    else:
        prev_cache_blocks = kv_split_index * num_cache_blocks_per_split
        num_cache_blocks_to_process = min(num_cache_blocks_per_split, current_seq_num_cache_blocks - prev_cache_blocks)

    # Offsets for each query vector in the split/group
    query_split_group_offsets = tl.arange(0, cxpr_query_chunk_size * cxpr_query_group_size_padded)
    # What query vector in the split?
    query_split_group_seq_offsets = this_query_split_offset + query_split_group_offsets // cxpr_query_group_size_padded
    # What query head in the group?
    query_split_group_head_offsets = (
        kv_head_index * query_group_size
    ) + query_split_group_offsets % cxpr_query_group_size_padded

    # Need to mask out if any of these tokens are out of bounds
    query_split_group_seq_mask = query_split_group_seq_offsets < this_query_length
    query_split_group_head_mask = query_split_group_head_offsets < (kv_head_index * query_group_size) + query_group_size

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    # Offsets for the queries in this block
    query_offsets = (
        this_query_start * query_batch_stride
        + query_split_group_seq_offsets[:, None] * query_batch_stride
        + query_split_group_head_offsets[:, None] * query_head_stride
        + head_offsets[None, :]
    )

    # Mask out query elements that are just for padding
    query_mask = query_split_group_seq_mask[:, None] & query_split_group_head_mask[:, None] & head_mask[None, :]

    # Only need causal masking if enabled and this program is not processing a decode
    needs_causal_mask = cxpr_is_causal and not is_pure_decode

    # Load queries
    query = tl.load(query_ptr + query_offsets, mask=query_mask, other=0.0)

    if cxpr_apply_fp8_scaling:
        q_scale = tl.load(q_scale_ptr)
        query = (query * q_scale).to(dtype)

    # Index/offset for the current kv_head in the key_cache and value_cache
    kv_head_index_offset = kv_head_index * kv_head_stride

    # Pointer arithmetic to get to the entry in the block_table for the current batch_index
    current_block_table_offset = batch_index * block_table_batch_stride
    current_block_table_ptr = block_table_ptr + current_block_table_offset

    # Scratchpad for output from this group of cache blocks
    output = tl.zeros([cxpr_query_chunk_size * cxpr_query_group_size_padded, cxpr_head_size_padded], dtype=dtype)
    # Keep running max of softmax numerator (scale * Q * K)
    m_i = tl.full([cxpr_query_chunk_size * cxpr_query_group_size_padded], -float("inf"), dtype=dtype)
    # Keep running denominator of softmax
    l_i = tl.full([cxpr_query_chunk_size * cxpr_query_group_size_padded], 0.0, dtype=dtype)

    # Iterate through the cache blocks that this kernel is assigned to
    for cache_block_index in range(
        starting_cache_block_index, starting_cache_block_index + num_cache_blocks_to_process
    ):
        # Calculate number of entries in this cache block (will be a value between 1 and cache_block_size)
        num_entries_in_cache_block = min(
            current_sequence_length - (cache_block_index * cxpr_cache_block_size), cxpr_cache_block_size
        )

        cache_block_offsets = tl.arange(0, cxpr_cache_block_size)
        cache_block_mask = cache_block_offsets < num_entries_in_cache_block

        # Offset from the block_table row for the current batch by the number of cache blocks
        current_cache_block_number_ptr = current_block_table_ptr + cache_block_index
        physical_cache_block_number = tl.load(current_cache_block_number_ptr)

        # Calculate address of current cache block
        kv_cache_block_index_offset = physical_cache_block_number * kv_page_stride

        # Load the key block as (cxpr_head_size_padded, cache_block_size)
        # Note: we're loading it transposed here
        key_block_offsets = (
            cache_block_offsets[None, :] * kv_cache_block_stride + kv_head_index_offset + head_offsets[:, None]
        )

        key_block_mask = head_mask[:, None] & cache_block_mask[None, :]

        key_block = tl.load(
            key_cache_ptr + kv_cache_block_index_offset + key_block_offsets,
            mask=key_block_mask,
            other=0.0,
        )

        if cxpr_apply_fp8_scaling:
            # Dequantize (multiply by scale factor)
            k_scale = tl.load(k_scale_ptr)
            key_block = (key_block * k_scale).to(dtype)

        # Multiply query vector by key matrix for this cache block (and apply scaling factor)
        # query.shape -> (query_chunk_size * query_group_size, head_size)
        # key_block.shape -> (head_size, cache_block_size)
        # qk.shape -> (query_chunk_size * query_group_size, cache_block_size)
        qk = (scale * tl.dot(query, key_block)).to(dtype)

        # Need to mask out any elements that represent unused cache block entries or padding elements
        qk_mask = query_split_group_seq_mask[:, None] & query_split_group_head_mask[:, None] & cache_block_mask[None, :]

        if needs_causal_mask:
            effective_seqlen_k_offsets = cache_block_index * cxpr_cache_block_size + cache_block_offsets
            causal_mask = query_split_group_seq_offsets[:, None] >= effective_seqlen_k_offsets[None, :]
            qk_mask = qk_mask & causal_mask

        qk = tl.where(qk_mask, qk, -float("inf")).to(dtype)

        # Handle softcapping
        if cxpr_is_softcap:
            # tanh can only accept fp32 or fp64 arguments
            qk = (softcap * libdevice.tanh((qk / softcap).to(tl.float32))).to(dtype)

        # Reduce maximum between running max and the max of (scale * Q * K) for this cache block
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1)).to(dtype)

        # Calculate numerator of softmax for this cache block
        p = tl.exp((qk - m_ij[:, None]).to(tl.float32)).to(dtype)
        # Need to mask out any elements that represent unused cache block entries or padding elements
        p = tl.where(qk_mask, p, 0.0).to(dtype)

        # Calculate sum of softmax numerator for this cache block
        l_ij = tl.sum(p, axis=1).to(dtype)

        # Calculate correction factor for this cache block
        alpha = tl.exp((m_i - m_ij).to(tl.float32)).to(dtype)

        # Apply scaling factor to running output
        output *= alpha[:, None]

        # Load the value block as (cache_block_size, cxpr_head_size_padded)
        value_block_offsets = (
            cache_block_offsets[:, None] * kv_cache_block_stride + kv_head_index_offset + head_offsets[None, :]
        )

        value_block_mask = cache_block_mask[:, None] & head_mask[None, :]

        value_block = tl.load(
            value_cache_ptr + kv_cache_block_index_offset + value_block_offsets,
            mask=value_block_mask,
            other=0.0,
        )

        if cxpr_apply_fp8_scaling:
            # Dequantize (multiply by scale factor)
            v_scale = tl.load(v_scale_ptr)
            value_block = (value_block * v_scale).to(dtype)

        # Multiply softmax probabilities by value matrix for this cache block
        # p.shape -> (query_chunk_size * query_group_size, cache_block_size)
        # value_block.shape -> (cache_block_size, head_size)
        # output.shape -> (query_chunk_size * query_group_size, head_size)
        output += tl.dot(p, value_block).to(dtype)

        # Update running max
        m_i = m_ij
        # Update running denominator
        l_i = l_i * alpha + l_ij

    # Apply correction for denominator of these cache blocks
    output /= l_i[:, None]

    # Calculate offsets to store the output for this query split/query group
    # 2D block of shape (query_chunk_size * query_group_size, head_size_padded)
    output_scratch_offsets = (
        kv_split_index * output_scratchpad_kv_split_stride
        + this_query_start * output_scratchpad_batch_stride
        + query_split_group_seq_offsets[:, None] * output_scratchpad_batch_stride
        + query_split_group_head_offsets[:, None] * output_scratchpad_head_stride
        + head_offsets[None, :]
    )

    # Store output scratchpad results
    tl.store(
        output_scratchpad_ptr + output_scratch_offsets,
        output,
        mask=query_mask,
    )

    if cxpr_split_kv:
        # Calculate scratchpad log(sum(exp))
        # Note: log() only accepts fp32/fp64 arguments
        lse = m_i + tl.log(l_i.to(tl.float32)).to(dtype)

        # Calculate offsets to store log-sum-exp for this query split/query group
        # 1D block of shape (query_chunk_size * query_group_size,)
        lse_scratch_offsets = (
            kv_split_index * lse_scratchpad_kv_split_stride
            + this_query_start * lse_scratchpad_batch_stride
            + query_split_group_seq_offsets * lse_scratchpad_batch_stride
            + query_split_group_head_offsets
        )

        lse_mask = query_split_group_seq_mask & query_split_group_head_mask

        # Store lse scratchpad results
        tl.store(
            lse_scratchpad_ptr + lse_scratch_offsets,
            lse,
            mask=lse_mask,
        )


@triton.jit  # type: ignore[misc]
def _varlen_attention_reduce_splits_kernel(  # noqa: PLR0913
    # Pointers to tensors
    output_ptr: tl.tensor,  # (total_num_q, num_query_heads, head_size)
    output_scratchpad_ptr: tl.tensor,  # (num_kv_splits, total_num_q, num_query_heads, head_size)
    lse_scratchpad_ptr: tl.tensor,  # (num_kv_splits, total_num_q, num_query_heads)
    seq_lens_ptr: tl.tensor,  # (batch_size, )
    # Scalars
    num_cache_blocks_per_split: int,
    # Sizes of tensors above
    head_size: int,  # output.shape[2]
    batch_size: int,
    # Strides for tensors above
    output_batch_stride: int,  # output.stride(0)
    output_head_stride: int,  # output.stride(1)
    output_scratchpad_kv_split_stride: int,  # output_scratchpad.stride(0)
    output_scratchpad_batch_stride: int,  # output_scratchpad.stride(1)
    output_scratchpad_head_stride: int,  # output_scratchpad.stride(2)
    lse_scratchpad_kv_split_stride: int,  # lse_scratchpad.stride(0)
    lse_scratchpad_batch_stride: int,  # lse_scratchpad.stride(1)
    # Constexprs
    cxpr_cache_block_size: tl.constexpr,
    cxpr_head_size_padded: tl.constexpr,
) -> None:
    """Varlen Attention kernel: reduce results across all splits.

    Args:
        output_ptr: Pointer to tensor for final output, shape: (total_num_q, num_query_heads, head_size).
        output_scratchpad_ptr: Pointer to tensor as scratchpad for output of each cache block, shape: (num_kv_splits, total_num_q, num_query_heads, head_size).
        lse_scratchpad_ptr: Pointer to tensor as scratchpad for log-sum-exp of each cache block, shape: (num_kv_splits, total_num_q, num_query_heads).
        seq_lens_ptr: Pointer to tensor holding the current sequence length for each sequence in the batch, shape: (batch_size, ).
        cu_seqlens_q_ptr: Pointer to tensor holding the cumulative sequence lengths for each query in the batch, shape: (batch_size + 1, ).
        num_cache_blocks_per_split: The maximum number of cache blocks each split will process.
        head_size: Actual head dim, not padded to power-of-two.
        batch_size: Number of sequences in the batch.
        output_batch_stride: Stride of the output tensor in the 0th dimension.
        output_head_stride: Stride of the output tensor in the 1st dimension.
        output_scratchpad_kv_split_stride: Stride of the output scratchpad tensor in the 0th dimension.
        output_scratchpad_batch_stride: Stride of the output scratchpad tensor in the 1st dimension.
        output_scratchpad_head_stride: Stride of the output scratchpad tensor in the 2nd dimension.
        lse_scratchpad_kv_split_stride: Stride of the log-sum-exp scratchpad tensor in the 0th dimension.
        lse_scratchpad_batch_stride: Stride of the log-sum-exp scratchpad tensor in the 1st dimension.
        cxpr_query_chunk_size: The size of the query chunks (must be power of two!).
        cxpr_cache_block_size: The size of the cache blocks (must be power of two!), as constexpr so that we can use for reshaping tensors.
        cxpr_head_size_padded: The head size of the attention layer padded to the next power of two.
        cxpr_is_causal: Whether or not to apply causal masking.
    """
    # What query head is this program processing?
    query_head_index = tl.program_id(0)
    # What batch is this program processing?
    batch_index = tl.program_id(1)

    # Get type that we should be using for accumulating results/intermediate calculations
    dtype = output_ptr.dtype.element_ty

    # Accumulator for the output of this batch/head
    output = tl.zeros([cxpr_head_size_padded], dtype=dtype)
    # Running max of block lse
    m_i = tl.full([], -float("inf"), dtype=dtype)
    # Running final scale factor
    l_i = tl.full([], 0.0, dtype=dtype)

    # Load scalar current_sequence_length for the current batch
    current_sequence_length = tl.load(seq_lens_ptr + batch_index)

    # The length of the current sequence will tell us how many cache blocks we need to read
    current_seq_num_cache_blocks = tl.cdiv(current_sequence_length, cxpr_cache_block_size)

    # How many KV splits do we need to process
    num_kv_splits_this_seq = tl.cdiv(current_seq_num_cache_blocks, num_cache_blocks_per_split)

    # Offsets to each element of the padded-to-next-power-of-two head size
    head_offsets = tl.arange(0, cxpr_head_size_padded)
    # Mask to only read valid indices of the actual head size
    head_mask = head_offsets < head_size

    # Iterate through every cache block for the current sequence
    for kv_split_index in range(num_kv_splits_this_seq):
        # Calculate offsets to load the scratch for this head/batch/split
        # Block of shape (cxpr_head_size_padded,)
        output_scratchpad_offsets = (
            kv_split_index * output_scratchpad_kv_split_stride
            + batch_index * output_scratchpad_batch_stride
            + query_head_index * output_scratchpad_head_stride
            + head_offsets
        )

        # Load output for this cache block, shape -> (cxpr_head_size_padded,)
        block_output = tl.load(
            output_scratchpad_ptr + output_scratchpad_offsets,
            mask=head_mask,
            other=0.0,
        )

        # Calculate offsets to load log-sum-exp for this head/batch/cache block
        lse_scratchpad_offsets = (
            kv_split_index * lse_scratchpad_kv_split_stride
            + batch_index * lse_scratchpad_batch_stride
            + query_head_index
        )

        # Load log-sum-exp for this cache block, shape -> scalar
        block_lse = tl.load(lse_scratchpad_ptr + lse_scratchpad_offsets)

        # Reduce running max lse
        m_ij = tl.maximum(m_i, block_lse).to(dtype)

        # Calculate correction factor from previous cache blocks
        # Note: exp() only accepts fp32/fp64 arguments
        alpha = tl.exp((m_i - m_ij).to(tl.float32)).to(dtype)

        # Apply correction factor
        output *= alpha

        # Calculate correction factor from this cache block
        # Note: exp() only accepts fp32/fp64 arguments
        beta = tl.exp((block_lse - m_ij).to(tl.float32)).to(dtype)

        # Apply second correction factor and accumulate running output
        output += (beta * block_output).to(dtype)

        # Update running max
        m_i = m_ij
        # Update running final scale factor
        l_i = l_i * alpha + beta

    # Apply final correction to output
    output /= l_i

    # Calculate offsets to store the output for this head/batch
    output_offsets = batch_index * output_batch_stride + query_head_index * output_head_stride + head_offsets

    # Store final result
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=head_mask,
    )


def _get_block_size() -> int:
    """Get block size for tuning purposes."""
    # Note: in the future we could specify different settings here based on the platform or autotune,
    # but for simplicity right now we'll hardcode this block size because it gives good performance
    # on A10, H100, and MI300X.
    return 32


def _get_tuned_sizes(head_size_padded: int, query_group_size_padded: int, max_seqlen_q: int) -> tuple[int, int]:
    """Get tuned sizes for current device."""
    block_size = _get_block_size()

    # If the head size grows too large then we want to limit how many queries we're chunking together
    small_head_size_limit: Final = 128
    if head_size_padded > small_head_size_limit:
        block_size = block_size // 2

    # The "query chunk size" represents the number of queries that each kernel will process at a time.
    query_chunk_size = max(1, block_size // query_group_size_padded) if max_seqlen_q > 1 else 1

    # If we have all decodes, we need to make sure that we have at least a block size of 16 for tl.dot
    query_group_size_padded = query_group_size_padded if max_seqlen_q > 1 else max(16, query_group_size_padded)

    return query_chunk_size, query_group_size_padded


def varlen_attention_launcher(  # noqa: PLR0913
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seq_lens: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    output_scratchpad: torch.Tensor | None = None,
    lse_scratchpad: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    strict: bool = False,
) -> None:
    """Varlen Attention kernel launcher.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        cu_seqlens_q: Tensor with the cumulative query sequence lengths for each index in the batch, shape: (batch_size + 1, ).
        max_seqlen_q: Maximum sequence length of the query.
        seq_lens: Tensor with the sequence length of each index in the batch, shape: (batch_size, ).
        max_seqlen_k: Maximum sequence length of the key/value.
        block_table: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        output_scratchpad: Tensor used as scratchpad to share cache block outputs between two stages, shape: (num_kv_splits, total_num_q, num_query_heads, head_size)
        lse_scratchpad: Tensor used as scratchpad to share cache block log-sum-exp between two stages, shape: (num_kv_splits, total_num_q, num_query_heads)
        causal: Whether or not to apply causal masking.
        scale: Scaling factor, 1/sqrt(head_size).
        softcap: Softcap value to apply to logits.
        kv_cache_dtype: Data type of the key/value cache.
        q_scale: Scaling factor for Q values.
        k_scale: Scaling factor for K values.
        v_scale: Scaling factor for V values.
        strict: (Optional) Whether to enable strict checking of input sizes.
    """
    if strict:
        assert query.shape == output.shape
        assert key_cache.shape == value_cache.shape
        assert key_cache.stride(0) == value_cache.stride(0)
        assert key_cache.stride(1) == value_cache.stride(1)
        assert key_cache.stride(2) == value_cache.stride(2)
        assert key_cache.stride(3) == value_cache.stride(3)
        assert key_cache.stride(3) == 1
        assert softcap >= 0.0

        expected_output_dtype = torch.bfloat16 if query.dtype in _FP8_DTYPES else query.dtype
        allowed_in_out_dtypes = [torch.float32, torch.float16, torch.bfloat16] + _FP8_DTYPES
        assert query.dtype in allowed_in_out_dtypes
        assert output.dtype == expected_output_dtype

        if output_scratchpad is not None or lse_scratchpad is not None:
            assert output_scratchpad is not None
            assert lse_scratchpad is not None
            assert output_scratchpad.dtype == expected_output_dtype
            assert lse_scratchpad.dtype == expected_output_dtype
            assert output_scratchpad.size(1) == lse_scratchpad.size(1)

    # Perform unchecked size accesses, assume has already been checked
    total_num_q, num_query_heads, head_size = output.shape
    num_cache_blocks, cache_block_size, num_kv_heads, _ = key_cache.shape
    batch_size, max_num_blocks_per_sequence = block_table.shape
    max_num_kv_splits = output_scratchpad.size(0) if output_scratchpad is not None else 1

    if strict:
        assert cache_block_size == triton.next_power_of_2(cache_block_size), "Cache block size must be a power of two!"

    # Need sizes to be constexpr in order to reshape tensors in kernel
    cxpr_cache_block_size: tl.constexpr = cache_block_size
    cxpr_head_size_padded: tl.constexpr = triton.next_power_of_2(head_size)

    # For parity with Dao Flash Attention, softcap == 0.0 means no softcapping
    cxpr_is_softcap: tl.constexpr = softcap > 0.0

    cxpr_apply_fp8_scaling: tl.constexpr = "fp8" in kv_cache_dtype or query.dtype in _FP8_DTYPES

    if strict and cxpr_apply_fp8_scaling:
        assert q_scale is not None
        assert k_scale is not None
        assert v_scale is not None
        assert q_scale.numel() == 1
        assert k_scale.numel() == 1
        assert v_scale.numel() == 1

    # How many query heads correspond to the same KV head?
    query_group_size = num_query_heads // num_kv_heads
    query_group_size_padded = triton.next_power_of_2(query_group_size)

    # What is the maximum number of stage 1 kernels to launch per batch/head?
    # Each kernel processes up to {cache_block_size} tokens at a time (for each split that a stage1 kernel processes), so we
    # can process a sequence up to {max_num_kv_splits * cache_block_size} tokens before a stage 1 kernel will process multiple
    # cache blocks. This helps to reduce the overhead of kernel launches / split reduction for long sequences.
    # Note: we may need to tune this value for a given HW platform.
    num_kv_splits = min(max_num_blocks_per_sequence, max_num_kv_splits)

    # Different platforms may require different sizes.
    query_chunk_size, query_group_size_padded = _get_tuned_sizes(
        cxpr_head_size_padded, query_group_size_padded, max_seqlen_q
    )

    # Use the maximum Q sequence length to determine what is the max number of query splits any sequence will need
    num_query_splits = triton.cdiv(max_seqlen_q, query_chunk_size)

    # How many cache blocks will each kernel process?
    num_cache_blocks_per_split = triton.cdiv(max_num_blocks_per_sequence, num_kv_splits)

    # Use default scaling factor if not provided
    if scale is None:
        scale = float(1.0 / (head_size**0.5))

    output_scratchpad_kv_split_stride = 0
    output_scratchpad_batch_stride = output.stride(0)
    output_scratchpad_head_stride = output.stride(1)
    lse_scratchpad_kv_split_stride = 0
    lse_scratchpad_batch_stride = 0

    if num_kv_splits > 1:
        assert output_scratchpad is not None
        assert lse_scratchpad is not None

        output_scratchpad_kv_split_stride = output_scratchpad.stride(0)
        output_scratchpad_batch_stride = output_scratchpad.stride(1)
        output_scratchpad_head_stride = output_scratchpad.stride(2)
        lse_scratchpad_kv_split_stride = lse_scratchpad.stride(0)
        lse_scratchpad_batch_stride = lse_scratchpad.stride(1)

    # For computing attention for split block (stage 1): parallelize over query splits, KV splits, KV heads, and batches.
    stage1_grid = (num_query_splits * num_kv_splits, num_kv_heads, batch_size)

    # Launch stage 1 kernel
    _varlen_attention_compute_splits_kernel[stage1_grid](
        # Relevant tensors
        output_scratchpad_ptr=output_scratchpad if num_kv_splits > 1 else output,
        lse_scratchpad_ptr=lse_scratchpad,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_table_ptr=block_table,
        seq_lens_ptr=seq_lens,
        cu_seqlens_q_ptr=cu_seqlens_q,
        q_scale_ptr=q_scale,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        # Scalars
        scale=scale,
        num_cache_blocks_per_split=num_cache_blocks_per_split,
        softcap=softcap,
        head_size=head_size,
        query_group_size=query_group_size,
        num_kv_splits=num_kv_splits,
        # Strides of relevant tensors
        output_scratchpad_kv_split_stride=output_scratchpad_kv_split_stride,
        output_scratchpad_batch_stride=output_scratchpad_batch_stride,
        output_scratchpad_head_stride=output_scratchpad_head_stride,
        lse_scratchpad_kv_split_stride=lse_scratchpad_kv_split_stride,
        lse_scratchpad_batch_stride=lse_scratchpad_batch_stride,
        query_batch_stride=query.stride(0),
        query_head_stride=query.stride(1),
        kv_page_stride=key_cache.stride(0),
        kv_cache_block_stride=key_cache.stride(1),
        kv_head_stride=key_cache.stride(2),
        kv_head_element_stride=key_cache.stride(3),
        block_table_batch_stride=block_table.stride(0),
        # Constexpr sizes
        cxpr_query_group_size_padded=query_group_size_padded,
        cxpr_query_chunk_size=query_chunk_size,
        cxpr_cache_block_size=cxpr_cache_block_size,
        cxpr_head_size_padded=cxpr_head_size_padded,
        cxpr_is_softcap=cxpr_is_softcap,
        cxpr_apply_fp8_scaling=cxpr_apply_fp8_scaling,
        cxpr_is_causal=causal,
        cxpr_split_kv=(num_kv_splits > 1),
    )

    if num_kv_splits > 1:
        assert output_scratchpad is not None
        assert lse_scratchpad is not None

        # For reducing over splits (stage 2): parallelize over query heads and batches
        stage2_grid = (num_query_heads, batch_size)

        # Launch stage 2 kernel
        _varlen_attention_reduce_splits_kernel[stage2_grid](
            # Relevant tensors
            output_ptr=output,
            output_scratchpad_ptr=output_scratchpad,
            lse_scratchpad_ptr=lse_scratchpad,
            seq_lens_ptr=seq_lens,
            # Scalars
            num_cache_blocks_per_split=num_cache_blocks_per_split,
            head_size=head_size,
            batch_size=batch_size,
            # Strides of relevant tensors
            output_batch_stride=output.stride(0),
            output_head_stride=output.stride(1),
            output_scratchpad_kv_split_stride=output_scratchpad_kv_split_stride,
            output_scratchpad_batch_stride=output_scratchpad_batch_stride,
            output_scratchpad_head_stride=output_scratchpad_head_stride,
            lse_scratchpad_kv_split_stride=lse_scratchpad_kv_split_stride,
            lse_scratchpad_batch_stride=lse_scratchpad_batch_stride,
            # Constexpr sizes
            cxpr_cache_block_size=cxpr_cache_block_size,
            cxpr_head_size_padded=cxpr_head_size_padded,
        )
