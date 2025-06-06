# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention with varlen."""

from dataclasses import dataclass
from typing import Final

import torch

from conch.kernels.attention.varlen_attention import varlen_attention_launcher


@dataclass
class VarlenAttentionMetadata:
    """Wrapper class holding metadata for variable-length attention kernel."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    total_num_q: int
    max_num_blocks_per_sequence: int
    num_kv_splits: int


def _check_output_query_size_compatibility(output: torch.Tensor, query: torch.Tensor) -> None:
    """Check size compatibility of Output and Query tensors.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).

    Raises:
        ValueError if sizes are mismatched.
    """
    # Output tensor should be a 3-D tensor of shape (total_num_q, num_query_heads, head_size)
    expected_output_shape_dims: Final = 3

    if len(output.shape) != expected_output_shape_dims:
        msg = f"Output tensor has unexpected shape ({output.shape = }), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    # Query tensor should have same shape as output
    if len(query.shape) != expected_output_shape_dims:
        msg = f"Query tensor has unexpected shape ({query.shape = }), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    if query.shape != output.shape:
        msg = f"Shape of query and output tensors does not match ({query.shape = }, {output.shape = })"
        raise ValueError(msg)


def _check_key_value_cache_size_compatibility(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    head_size: int,
    num_query_heads: int,
) -> None:
    """Check size compatibility of Key and Value tensors.

    Args:
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        head_size: Size of attention head, deduced from query tensor size.
        num_query_heads: Number of query heads, deduced from query tensor size.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value tensors should be 4-D tensors of shape (num_blocks, num_kv_heads, cache_block_size, head_size)
    expected_key_shape_dims: Final = 4

    if len(key_cache.shape) != expected_key_shape_dims:
        msg = (
            f"key_cache tensor has unexpected shape ({key_cache.shape = }), expected {expected_key_shape_dims}-D tensor"
        )
        raise ValueError(msg)

    if len(value_cache.shape) != expected_key_shape_dims:
        msg = f"value_cache tensor has unexpected shape ({value_cache.shape = }), expected {expected_key_shape_dims}-D tensor"
        raise ValueError(msg)

    if key_cache.shape != value_cache.shape:
        msg = (
            f"Shape of key_cache and value_cache tensors does not match ({key_cache.shape = }, {value_cache.shape = })"
        )
        raise ValueError(msg)

    num_blocks, cache_block_size, num_kv_heads, head_size_kv = key_cache.shape

    if head_size_kv != head_size:
        msg = f"Head size of key/value cache tensors does not match head size of query/output tensors ({head_size_kv = }, {head_size = })"
        raise ValueError(msg)

    if num_kv_heads > num_query_heads:
        msg = f"Number of key/value heads ({num_kv_heads}) is greater than number of query heads ({num_query_heads})"
        raise ValueError(msg)


def _check_cumulative_sequence_length_size_compatibility(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor
) -> None:
    """Check size compatibility of cumulative sequence length tensors.

    Args:
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        cu_seqlens_k: Cumulative sequence length for key/value tensors, shape: (batch_size + 1).

    Raises:
        ValueError if sizes are mismatched.
    """
    # Cumulative sequence length tensors should be 1-D tensors of shape (batch_size + 1)
    expected_cu_seqlen_shape_dims: Final = 1

    if len(cu_seqlens_q.shape) != expected_cu_seqlen_shape_dims:
        msg = f"Cumulative sequence length tensor for query has unexpected shape ({cu_seqlens_q.shape = }), expected {expected_cu_seqlen_shape_dims}-D tensor"
        raise ValueError(msg)

    if len(cu_seqlens_k.shape) != expected_cu_seqlen_shape_dims:
        msg = f"Cumulative sequence length tensor for key has unexpected shape ({cu_seqlens_k.shape = }), expected {expected_cu_seqlen_shape_dims}-D tensor"
        raise ValueError(msg)

    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        msg = f"Shape of cumulative sequence length tensors does not match ({cu_seqlens_q.shape = }, {cu_seqlens_k.shape = })"
        raise ValueError(msg)


def _check_block_table_size_compatibility(block_table: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of block_table tensor.

    Args:
        block_table: Block tables tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    batch_size_block_table: int = block_table.shape[0]

    if batch_size_block_table != batch_size:
        msg = f"Batch size from block_table tensor ({batch_size_block_table}) does not match batch_size from output/query tensors ({batch_size})"
        raise ValueError(msg)


def _check_seqlen_size_compatibility(seq_lens: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of seq_lens tensor.

    Args:
        seq_lens: Sequence lengths tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Sequence lengths tensor should be 1-D tensors of shape (batch_size,)
    expected_seqlen_shape_dims: Final = 1

    if len(seq_lens.shape) != expected_seqlen_shape_dims:
        msg = f"Sequence lengths tensor has unexpected shape ({seq_lens.shape = }), expected {expected_seqlen_shape_dims}-D tensor"
        raise ValueError(msg)

    if seq_lens.shape[0] != batch_size:
        msg = f"Shape of sequence lengths tensor does not match batch size ({seq_lens.shape[0] = }, {batch_size = })"
        raise ValueError(msg)


def _determine_max_num_kv_splits(max_seqlen_q: int, max_seqlen_k: int) -> int:
    # If we have any prefills, disable FlashDecoding/KV-splits
    if max_seqlen_q > 1:
        return 1

    # Note: this is a pretty basic heuristic, we could try and do something more sophisticated in the future
    if max_seqlen_k > 8192:
        return 64

    if max_seqlen_k > 2048:
        return 32

    if max_seqlen_k > 1024:
        return 16

    if max_seqlen_k > 512:
        return 8

    if max_seqlen_k > 256:
        return 4

    if max_seqlen_k > 128:
        return 2

    return 1


def _create_varlen_metadata(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> VarlenAttentionMetadata:
    """Check size compatibility of tensors for variable-length attention and return metadata if successful.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        cu_seqlens_k: Cumulative sequence length for key/value tensors, shape: (batch_size + 1).
        block_table: Block tables tensor, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Sequence lengths tensor, shape: (batch_size,).

    Raises:
        ValueError if sizes are mismatched.

    Returns:
        Metadata dataclass holding information about tensor sizes.
    """
    _check_output_query_size_compatibility(out, query)
    total_num_q, num_query_heads, head_size = out.shape

    _check_key_value_cache_size_compatibility(key_cache, value_cache, head_size, num_query_heads)
    _, _, num_kv_heads, _ = key_cache.shape

    _check_cumulative_sequence_length_size_compatibility(cu_seqlens_q, cu_seqlens_k)
    batch_size = cu_seqlens_q.shape[0] - 1

    _check_block_table_size_compatibility(block_table, batch_size)
    _, max_num_blocks_per_sequence = block_table.shape

    _check_seqlen_size_compatibility(seq_lens, batch_size)

    return VarlenAttentionMetadata(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        total_num_q=total_num_q,
        max_num_blocks_per_sequence=max_num_blocks_per_sequence,
        num_kv_splits=_determine_max_num_kv_splits(max_seqlen_q, max_seqlen_k),
    )


def varlen_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    output: torch.Tensor | None = None,
    causal: bool = False,
    scale: float | None = None,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Varlen attention interface to verify sizes and launch kernel.

    Args:
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key_cache: Tensor with cached K values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_cache: Tensor with cached V values, shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_table: Block tables tensor, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Sequence lengths tensor, shape: (batch_size,).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        cu_seqlens_k: Cumulative sequence length for key/value tensors, shape: (batch_size + 1).
        max_seqlen_q: Maximum sequence length for query/output tensors.
        max_seqlen_k: Maximum sequence length for key/value tensors.
        output: (Optional), Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        causal: (Optional), Whether to apply causal masking.
        scale: (Optional), Scaling factor, 1/sqrt(head_size).
        softcap: (Optional), Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: (Optional), String datatype of KV-cache.
        k_scale: (Optional), Scaling factor for K values.
        v_scale: (Optional), Scaling factor for V values.
    """
    # Allocate output tensor if not provided
    if output is None:
        output = torch.zeros_like(query, device=query.device, dtype=query.dtype)

    # Check sizes of input tensors
    metadata = _create_varlen_metadata(
        output,
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        block_table,
        seq_lens,
        max_seqlen_q,
        max_seqlen_k,
    )

    output_scratchpad = None
    lse_scratchpad = None

    if metadata.num_kv_splits > 1:
        # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/kv split/query head
        output_scratchpad = torch.zeros(
            (metadata.total_num_q, metadata.num_kv_splits, metadata.num_query_heads, metadata.head_size),
            dtype=output.dtype,
            device=output.device,
        )

        # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/kv split/query head
        lse_scratchpad = torch.zeros(
            (metadata.total_num_q, metadata.num_kv_splits, metadata.num_query_heads),
            dtype=output.dtype,
            device=output.device,
        )

    varlen_attention_launcher(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        output_scratchpad=output_scratchpad,
        lse_scratchpad=lse_scratchpad,
        causal=causal,
        scale=scale,
        softcap=softcap,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    return output
