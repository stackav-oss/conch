# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation of rotary embedding kernel."""

import torch

from conch import envs
from conch.platforms import current_platform


def _compute_inv_freq(base: float, rotary_dim: int) -> torch.Tensor:
    """Compute the inverse frequency."""
    return 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  # type: ignore[no-any-return]


def compute_cos_sin_cache(base: float, rotary_dim: int, max_position_embeddings: int) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = _compute_inv_freq(base, rotary_dim)
    t = torch.arange(max_position_embeddings, dtype=torch.float)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def _rotary_embedding_pytorch_ref(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    *,
    is_neox_style: bool = True,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference rotary_embedding impl."""
    if offsets is not None:
        positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, is_neox_style=is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = _apply_rotary_emb(key_rot, cos, sin, is_neox_style=is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _rotary_embedding_vllm_ref(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    *,
    is_neox_style: bool = True,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM reference rotary_embedding impl."""
    from vllm import _custom_ops as vllm_custom_ops

    cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)

    # vllm_custom_ops.rotary_embedding()/batched_rotary_embedding()
    # are in-place operations that update the query and key tensors.
    if offsets is not None:
        vllm_custom_ops.batched_rotary_embedding(
            positions, query, key, head_size, cos_sin_cache, is_neox_style, rotary_dim, offsets
        )
    else:
        vllm_custom_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)

    return query, key


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    *,
    is_neox_style: bool = True,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _rotary_embedding_vllm_ref(
            positions,
            query,
            key,
            cos_sin_cache,
            rotary_dim,
            head_size,
            is_neox_style=is_neox_style,
            offsets=offsets,
        )

    return _rotary_embedding_pytorch_ref(
        positions,
        query,
        key,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style=is_neox_style,
        offsets=offsets,
    )
