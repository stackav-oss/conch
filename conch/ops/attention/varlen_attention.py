# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Flash Attention & Varlen."""

from dataclasses import dataclass
from typing import Final

import torch

from conch.kernels.attention.varlen_attention import varlen_attention_launcher, MAX_NUM_KV_SPLITS


@dataclass
class VarlenAttentionMetadata:
    """Wrapper class holding metadata for variable-length attention kernel."""

    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    total_num_q: int
    total_num_k: int


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


def _check_key_value_size_compatibility(
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Check size compatibility of Key and Value tensors.

    Args:
        key: Key tensor, shape: (total_num_k, num_kv_heads, head_size).
        value: Value tensor, shape: (total_num_k, num_kv_heads, head_size).

    Raises:
        ValueError if sizes are mismatched.
    """
    # Key/Value tensors should be 3-D tensors of shape (total_num_k, num_kv_heads, head_size)
    expected_key_shape_dims: Final = 3

    if len(key.shape) != expected_key_shape_dims:
        msg = f"Key tensor has unexpected shape ({key.shape = }), expected {expected_key_shape_dims}-D tensor"
        raise ValueError(msg)

    if len(value.shape) != expected_key_shape_dims:
        msg = f"Value tensor has unexpected shape ({value.shape = }), expected {expected_key_shape_dims}-D tensor"
        raise ValueError(msg)

    if key.shape != value.shape:
        msg = f"Shape of key and value tensors does not match ({key.shape = }, {value.shape = })"
        raise ValueError(msg)


def _check_cumulative_sequence_length_size_compatibility(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> None:
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


def _check_size_compatibility(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> VarlenAttentionMetadata:
    """Check size compatibility of tensors for variable-length attention and return metadata if successful.

    Args:
        output: Tensor to write the output of the attention calculation, shape: (total_num_q, num_query_heads, head_size).
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key: Key tensor, shape: (total_num_k, num_kv_heads, head_size).
        value: Value tensor, shape: (total_num_k, num_kv_heads, head_size).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        cu_seqlens_k: Cumulative sequence length for key/value tensors, shape: (batch_size + 1).

    Raises:
        ValueError if sizes are mismatched.

    Returns:
        Metadata dataclass holding information about tensor sizes.
    """
    _check_output_query_size_compatibility(out, query)
    total_num_q, num_query_heads, head_size = out.shape

    _check_key_value_size_compatibility(key, value)
    total_num_k, num_kv_heads, _ = key.shape

    _check_cumulative_sequence_length_size_compatibility(cu_seqlens_q, cu_seqlens_k)
    batch_size = cu_seqlens_q.shape[0] - 1

    return VarlenAttentionMetadata(
        batch_size=batch_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        total_num_q=total_num_q,
        total_num_k=total_num_k,
    )


def varlen_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    softcap: float = 0.0,
) -> torch.Tensor:
    """Varlen attention interface to verify sizes and launch kernel.

    Args:
        query: Query tensor, shape: (total_num_q, num_query_heads, head_size).
        key_cache: Key tensor, shape: (total_num_k, num_kv_heads, head_size).
        value_cache: Value tensor, shape: (total_num_k, num_kv_heads, head_size).
        cu_seqlens_q: Cumulative sequence length for query/output tensors, shape: (batch_size + 1).
        cu_seqlens_k: Cumulative sequence length for key/value tensors, shape: (batch_size + 1).
        max_seqlen_q: Maximum sequence length for query/output tensors.
        max_seqlen_k: Maximum sequence length for key/value tensors.
        scale: Scaling factor, 1/sqrt(head_size).
        softcap: (Optional), Logit softcap to apply (0.0 means no softcap will be applied).
    """
    output = torch.zeros_like(query, device=query.device, dtype=query.dtype)

    batch_size = cu_seqlens_q[-1].item()

    # print(f"{batch_size = }")
    # return

    total_num_q, num_query_heads, head_size = query.shape

    # Check sizes of input tensors
    # _ = _check_size_compatibility(output, query, key_cache, value_cache, cu_seqlens_q, cu_seqlens_k)

    # Note: we could allocate this scratch outside of this function so that we could reuse it. vLLM allocates the scratch memory inline like this

    # Allocate additional memory for intermediate result (of shape (head_size,)) for each batch/query head/cache block
    output_scratchpad = torch.zeros(
        # (metadata.batch_size, MAX_NUM_SPLITS, metadata.num_query_heads, metadata.head_size),
        (batch_size, MAX_NUM_KV_SPLITS, num_query_heads, head_size),
        dtype=output.dtype,
        device=output.device,
    )

    # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/query head/cache block
    lse_scratchpad = torch.zeros(
        # (metadata.batch_size, MAX_NUM_SPLITS, metadata.num_query_heads),
        (batch_size, MAX_NUM_KV_SPLITS, num_query_heads),
        dtype=output.dtype,
        device=output.device,
    )

    varlen_attention_launcher(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output_scratchpad=output_scratchpad,
        lse_scratchpad=lse_scratchpad,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        softcap=softcap,
        kv_cache_dtype="auto",
    )

    return output
