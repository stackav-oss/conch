# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

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
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Reference PyTorch-only implementation of reshape_and_cache."""
    num_tokens, _, _ = key.shape
    _, _, _, block_size, _ = key_cache.shape

    if kv_cache_dtype == "fp8":
        k_scale = 1.0 / k_scale
        v_scale = 1.0 / v_scale
        fp8_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn

    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        if kv_cache_dtype == "fp8":
            key_cache[block_idx, :, :, block_offset, :] = (reshaped_key[i] * k_scale).to(fp8_dtype).view(torch.uint8)
            value_cache[block_idx, :, :, block_offset] = (value[i] * v_scale).to(fp8_dtype).view(torch.uint8)
        else:
            key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
            value_cache[block_idx, :, :, block_offset] = value[i]


def _reshape_and_cache_vllm_ref(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Reference vLLM implementation of reshape_and_cache."""
    from vllm._custom_ops import reshape_and_cache as reshape_and_cache_vllm

    reshape_and_cache_vllm(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
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
