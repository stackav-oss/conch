# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Flash Attention w/ Paged KV Cache + FlashDecoding."""

from dataclasses import dataclass
from typing import Final

import torch

from conch.kernels.attention.paged_attention import MAX_NUM_SPLITS, paged_attention_launcher


@dataclass
class PagedAttentionMetadata:
    """Wrapper class holding metadata for PagedAttention kernel."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    num_cache_blocks: int
    max_num_blocks_per_sequence: int


def _check_output_query_size_compatibility(out: torch.Tensor, query: torch.Tensor) -> None:
    """Check size compatibility of Output and Query tensors.

    Args:
        out: Output tensor.
        query: Query tensor.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Output tensor should be a 3-D tensor of shape (batch_size, num_heads, head_size)
    expected_output_shape_dims: Final = 3

    if len(out_shape := out.shape) != expected_output_shape_dims:
        msg = f"Output tensor has unexpected shape (shape={out_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    # Query tensor should have same shape as output
    if len(query_shape := query.shape) != expected_output_shape_dims:
        msg = f"Query tensor has unexpected shape (shape={query_shape}), expected {expected_output_shape_dims}-D tensor"
        raise ValueError(msg)

    if query_shape != out_shape:
        msg = f"Shape of query and output tensors does not match (query.shape={query_shape}, out.shape={out_shape})"
        raise ValueError(msg)


def _check_kv_cache_size_compatibility(
    kv_cache: torch.Tensor, num_kv_heads: int, cache_block_size: int, head_size: int
) -> None:
    """Check size compatibility of KV cache tensor.

    Args:
        kv_cache: KV cache tensor.
        num_kv_heads: Expected number of kv heads.
        cache_block_size: Expected size of cache block.
        head_size: Size of attention head, deduced from out/query tensor sizes.

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value Cache tensor should be a 3-D tensor of shape (2, num_blocks, cache_block_size * num_kv_heads * head_size)
    expected_kv_cache_shape_dims: Final = 3

    if len(kv_cache_shape := kv_cache.shape) != expected_kv_cache_shape_dims:
        msg = f"kv_cache tensor has unexpected shape (shape={kv_cache_shape}), expected {expected_kv_cache_shape_dims}-D tensor"
        raise ValueError(msg)

    expected_cache_line_size: Final = cache_block_size * num_kv_heads * head_size
    if kv_cache_shape[2] != expected_cache_line_size:
        msg = f"kv_cache line size ({kv_cache_shape[2]}) does not match expected ({expected_cache_line_size})"
        raise ValueError(msg)


def _check_block_table_size_compatibility(block_tables: torch.Tensor, batch_size: int) -> None:
    """Check size compatibility of block_tables tensor.

    Args:
        block_tables: Block tables tensor.
        batch_size: Expected size of batch.

    Raises:
        ValueError if sizes are mismatched.
    """
    batch_size_block_tables: int = block_tables.shape[0]

    if batch_size_block_tables != batch_size:
        msg = f"Batch size from block_tables tensor ({batch_size_block_tables}) does not match batch_size from output/query tensors ({batch_size})"
        raise ValueError(msg)


def _check_size_compatibility(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    num_kv_heads: int,
    cache_block_size: int,
) -> PagedAttentionMetadata:
    """Check size compatibility of tensors for PagedAttention and return Metadata if successful.

    Args:
        out: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        kv_cache: Combined KV cache tensor, shape: (2, num_blocks, cache_block_size * num_kv_heads * head_size).
        block_tables: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        num_kv_heads: The number of KV heads.
        cache_block_size: Size of the cache block.

    Raises:
        ValueError if sizes are mismatched.

    Returns:
        Metadata dataclass holding information about tensor sizes.
    """
    _check_output_query_size_compatibility(out, query)
    batch_size, num_query_heads, head_size = out.shape

    _check_kv_cache_size_compatibility(kv_cache, num_kv_heads, cache_block_size, head_size)
    _, num_cache_blocks, _ = kv_cache.shape

    _check_block_table_size_compatibility(block_tables, batch_size)
    _, max_num_blocks_per_sequence = block_tables.shape

    return PagedAttentionMetadata(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_cache_blocks=num_cache_blocks,
        max_num_blocks_per_sequence=max_num_blocks_per_sequence,
    )


def split_kv_cache(kv_cache: torch.Tensor, num_kv_heads: int, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split KV cache tensor into key_cache and value_cache.

    Args:
        kv_cache: Combined KV cache tensor, shape: (2, num_blocks, cache_block_size * num_kv_heads * head_size).
        num_kv_heads: Number of KV heads.
        head_size: Head size/dimension.

    Returns:
        Tuple of tensors, (key_cache, value_cache).
    """
    num_blocks: int = kv_cache.shape[1]

    key_cache = kv_cache[0]
    key_cache = key_cache.view(num_blocks, num_kv_heads, -1, head_size)

    value_cache = kv_cache[1]
    value_cache = value_cache.view(num_blocks, num_kv_heads, -1, head_size)

    return key_cache, value_cache


def paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_block_size: int,
    softcap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """PagedAttention interface to verify sizes, split kv_cache, and launch kernel.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (batch_size, num_heads, head_size).
        query: Query tensor, shape: (batch_size, num_heads, head_size).
        kv_cache: Tensor holding cache for K and V values, shape: (2, num_blocks, cache_block_size * num_kv_heads * head_size).
        num_kv_heads: The number of KV heads.
        scale: Scaling factor, 1/sqrt(head_size).
        block_tables: Tensor storing the mapping from batch to cache blocks, shape: (batch_size, max_num_blocks_per_sequence).
        seq_lens: Tensor with the sequence length of each index in the batch, shape: (batch_size, ).
        cache_block_size: Size of the cache block.
        softcap: (Optional), Logit softcap to apply (0.0 means no softcap will be applied).
        kv_cache_dtype: (Optional) Data type of the KV cache.
        k_scale: (Optional) FP8 scaling factor for key cache.
        v_scale: (Optional) FP8 scaling factor for value cache.
    """
    # Check sizes of input tensors
    metadata = _check_size_compatibility(output, query, kv_cache, block_tables, num_kv_heads, cache_block_size)

    key_cache, value_cache = split_kv_cache(kv_cache, num_kv_heads, metadata.head_size)

    # Note: we could allocate this scratch outside of this function so that we could reuse it. vLLM allocates the scratch memory inline like this

    # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/query head/cache block
    output_scratchpad = torch.zeros(
        (metadata.batch_size, MAX_NUM_SPLITS, metadata.num_query_heads, metadata.head_size),
        dtype=output.dtype,
        device=output.device,
    )

    # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/query head/cache block
    lse_scratchpad = torch.zeros(
        (metadata.batch_size, MAX_NUM_SPLITS, metadata.num_query_heads),
        dtype=output.dtype,
        device=output.device,
    )

    paged_attention_launcher(
        output,
        query,
        key_cache,
        value_cache,
        output_scratchpad,
        lse_scratchpad,
        scale,
        block_tables,
        seq_lens,
        softcap,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
