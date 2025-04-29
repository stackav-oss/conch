# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""vLLM reshape_and_cache."""

from typing import Final

import torch

from conch.kernels.vllm.reshape_and_cache import reshape_and_cache_launcher


def _validate_sizes(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Validate sizes of input tensors.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        value_cache: Value cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).

    Raises:
        ValueError if sizes are mismatched between various input tensors.
    """
    if key.shape != value.shape:
        msg = f"key.shape ({key.shape}) does not match value.shape ({value.shape})"
        raise ValueError(msg)

    expected_kv_dims: Final = 3
    if (key_dims := len(key.shape)) != expected_kv_dims:
        msg = f"Number of dimensions in key ({key_dims}) did not match expected ({expected_kv_dims})"
        raise ValueError(msg)

    num_tokens_kv, num_kv_heads_kv, head_size_kv = key.shape

    if key_cache.shape != value_cache.shape:
        msg = f"key_cache.shape ({key_cache.shape}) does not match value_cache.shape ({value_cache.shape})"
        raise ValueError(msg)

    expected_kv_cache_dims: Final = 4
    if (key_cache_dims := len(key_cache.shape)) != expected_kv_cache_dims:
        msg = f"Number of dimensions in key cache ({key_cache_dims}) did not match expected ({expected_kv_cache_dims})"
        raise ValueError(msg)

    _, num_kv_heads_kvc, _, head_size_kvc = key_cache.shape

    if num_kv_heads_kv != num_kv_heads_kvc:
        msg = f"Number of kv heads in key/value tensors ({num_kv_heads_kv}) does not match number of kv heads in key/value cache tensors ({num_kv_heads_kvc})"
        raise ValueError(msg)

    if head_size_kv != head_size_kvc:
        msg = f"Head size in key/value tensors ({head_size_kv}) does not match head size in key/value cache tensors ({head_size_kvc})"
        raise ValueError(msg)

    expected_slot_mapping_dims: Final = 1
    if (slot_mapping_dims := len(slot_mapping.shape)) != expected_slot_mapping_dims:
        msg = f"Number of dimensions in slot mapping ({slot_mapping_dims}) did not match expected ({expected_slot_mapping_dims})"
        raise ValueError(msg)

    num_tokens_sm = slot_mapping.size(0)

    if num_tokens_kv != num_tokens_sm:
        msg = f"Number of tokens in key/value tensors ({num_tokens_kv}) does not match number of tokens in slot mapping tensor ({num_tokens_sm})"
        raise ValueError(msg)


def _validate_kv_cache_dtype(kv_cache_dtype: str) -> bool:
    """Validate that KV Cache Dtype is valid and return whether to enable FP8 scaling.

    Args:
        kv_cache_dtype: String representing desired datatype of KV-cache.

    Returns:
        True if requested data type is FP8 and therefore _reshape_and_cache_kernel should apply k/v_scale.
    """
    fp8_dtypes: Final = {"fp8", "fp8_e4m3"}
    allowed_dtypes: Final = {"auto"}.union(fp8_dtypes)

    if kv_cache_dtype not in allowed_dtypes:
        msg = f"Unsupported kv_cache_dtype: '{kv_cache_dtype}'"
        raise ValueError(msg)

    return kv_cache_dtype in fp8_dtypes


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
    """Reshape key/value vectors and add them to the cache.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        value_cache: Value cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        kv_cache_dtype: String datatype of kv cache elements.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    # Verify input sizes/tensor shapes
    _validate_sizes(key, value, key_cache, value_cache, slot_mapping)

    # Validate kv cache dtype is valid and determine if k/v scaling factors must be applied
    apply_fp8_scaling = _validate_kv_cache_dtype(kv_cache_dtype)

    # Call kernel launch wrapper
    reshape_and_cache_launcher(
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale, apply_fp8_scaling=apply_fp8_scaling
    )
