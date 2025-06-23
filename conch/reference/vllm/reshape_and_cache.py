# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference implementation of vLLM reshape_and_cache."""

import torch

from conch import envs
from conch.platforms import current_platform


def _reshape_and_cache_pytorch_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Reference PyTorch-only implementation of reshape_and_cache."""
    num_tokens = slot_mapping.size(0)
    _, block_size, _, _ = key_cache.shape

    if kv_cache_dtype == "fp8":
        fp8_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn
        key = (key / k_scale).to(fp8_dtype).view(key_cache.dtype)
        value = (value / v_scale).to(fp8_dtype).view(value_cache.dtype)

    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size
    key_cache[block_indicies, block_offsets, :, :] = key[:num_tokens]
    value_cache[block_indicies, block_offsets, :, :] = value[:num_tokens]


def _reshape_and_cache_vllm_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Reference vLLM implementation of reshape_and_cache."""
    from vllm._custom_ops import reshape_and_cache_flash as reshape_and_cache_vllm

    reshape_and_cache_vllm(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Reference implementation of vLLM's reshape_and_cache operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _reshape_and_cache_vllm_ref(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    return _reshape_and_cache_pytorch_ref(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
