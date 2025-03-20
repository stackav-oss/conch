# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton implementation of vLLM reshape_and_cache_kernel."""

import torch
import triton
import triton.language as tl

from conch.platforms import current_platform


@triton.jit  # type: ignore[misc]
def _reshape_and_cache_kernel(
    # Pointers to tensors
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    key_cache_ptr: tl.tensor,
    value_cache_ptr: tl.tensor,
    slot_mapping_ptr: tl.tensor,
    # Strides of relevant tensors
    kv_token_stride: int,
    kv_head_stride: int,
    kv_cache_block_stride: int,
    kv_cache_head_stride: int,
    # Scalars
    cache_block_size: int,
    k_scale: float,
    v_scale: float,
    # Constexprs
    cxpr_head_size: tl.constexpr,
    cxpr_apply_fp8_scaling: tl.constexpr,
    cxpr_is_rocm: tl.constexpr,
) -> None:
    """Implementation of reshape_and_cache kernel.

    Args:
        key_ptr: Pointer to tensor of new key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value_ptr: Pointer to tensor of new value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache_ptr: Pointer to tensor of key cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        value_cache_ptr: Pointer to tensor of value cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        slot_mapping_ptr: Pointer to slot mapping tensor, shape: (num_tokens,).
        kv_token_stride: Stride of key/value tensors in 0th dimension.
        kv_head_stride: Stride of key/value tensors in 1st dimension.
        kv_cache_block_stride: Stride of key/value cache tensors in 0th dimension.
        kv_cache_head_stride: Stride of key/value cache tensors in 1st dimension.
        cache_block_size: Size of each cache block / page in the KV cache.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
        cxpr_head_size: Head size / dimension for the attention head (must be power of two!).
        cxpr_apply_fp8_scaling: Whether or not to apply FP8 scaling.
        cxpr_is_rocm: Whether or not we're on AMD.
    """
    # What token is this program processing?
    token_index = tl.program_id(0)
    # What head is this program processing?
    head_index = tl.program_id(1)

    # Get index of slot for this token from mapping tensor
    slot_index = tl.load(slot_mapping_ptr + token_index)

    # If slot index is negative its a padding token that should be ignored
    if slot_index < 0:
        return

    # Calculate index of cache block for this slot (value in range(0, num_cache_blocks))
    cache_block_index = slot_index // cache_block_size
    # Calculate entry index inside of a cache block/page for this slot (value in range(0, cache_block_size))
    entry_index = slot_index % cache_block_size

    # Calculate offset into key/value tensors to get to the token for this program
    kv_token_offset = token_index * kv_token_stride
    # Calculate offset into key/value tensors to get to the head for this program
    kv_head_offset = head_index * kv_head_stride
    # Offsets for each element of the head
    kv_head_offsets = tl.arange(0, cxpr_head_size)

    # Load key/value vectors for this token/head
    key = tl.load(key_ptr + kv_token_offset + kv_head_offset + kv_head_offsets)
    value = tl.load(value_ptr + kv_token_offset + kv_head_offset + kv_head_offsets)

    # Apply FP8 scaling if necessary
    if cxpr_apply_fp8_scaling:
        # We are quantizing these values, so multiply by the inverted scaling factor (inverted in launcher)
        key *= k_scale
        value *= v_scale

        fp8_dtype = tl.float8e4b8 if cxpr_is_rocm else tl.float8e4nv

        # First, cast to the lower-precision floating point type, then bitcast the fp8 representation
        # to uint8 to match the dtype of the cache
        key = key.to(fp8_dtype).to(key_cache_ptr.dtype.element_ty, bitcast=True)
        value = value.to(fp8_dtype).to(value_cache_ptr.dtype.element_ty, bitcast=True)

    # Calculate offset into key/value cache tensors to get to the cache block we're copying into
    kv_cache_block_offset = cache_block_index * kv_cache_block_stride
    # Calculate offset into key/value cache tensors to get to the head we're copying into
    kv_cache_head_offset = head_index * kv_cache_head_stride
    # Calculate offset in a cache block to get to the entry for we're copying into
    kv_cache_entry_offset = entry_index * cxpr_head_size

    # Store key/value vectors into cache
    tl.store(
        key_cache_ptr + kv_cache_block_offset + kv_cache_head_offset + kv_cache_entry_offset + kv_head_offsets, key
    )
    tl.store(
        value_cache_ptr + kv_cache_block_offset + kv_cache_head_offset + kv_cache_entry_offset + kv_head_offsets, value
    )


def reshape_and_cache_launcher(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: float,
    v_scale: float,
    *,
    apply_fp8_scaling: bool = False,
) -> None:
    """Launch reshape_and_cache kernel.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        value_cache: Value cache, shape: (num_cache_blocks, num_kv_heads, cache_block_size, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
        apply_fp8_scaling: Whether or not to apply fp8 scaling factors in kernel.
    """
    # Assume sizes already checked if calling launcher. For interface with strict size checking, call `reshape_and_cache()`.
    num_tokens, num_kv_heads, head_size = key.shape
    num_cache_blocks, _, cache_block_size, _ = key_cache.shape

    assert key.stride(0) == value.stride(0)  # noqa: S101
    assert key.stride(1) == value.stride(1)  # noqa: S101
    assert key.stride(2) == value.stride(2)  # noqa: S101
    assert key.stride(2) == 1  # noqa: S101

    assert key_cache.stride(0) == value_cache.stride(0)  # noqa: S101
    assert key_cache.stride(1) == value_cache.stride(1)  # noqa: S101
    assert key_cache.stride(2) == value_cache.stride(2)  # noqa: S101
    assert key_cache.stride(3) == value_cache.stride(3)  # noqa: S101
    assert key_cache.stride(3) == 1  # noqa: S101

    assert cache_block_size == triton.next_power_of_2(cache_block_size), "Cache block size must be a power of two!"  # noqa: S101
    assert head_size == triton.next_power_of_2(head_size), "Head size must be a power of two!"  # noqa: S101

    is_rocm: tl.constexpr = current_platform.is_amd()

    # Invert scale factors
    k_scale = 1.0 / k_scale
    v_scale = 1.0 / v_scale

    # Parallelize over the number of tokens and number of kv heads
    grid = (num_tokens, num_kv_heads)

    # Launch kernel
    _reshape_and_cache_kernel[grid](
        # Tensors
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        # Strides of relevant tensors
        key.stride(0),
        key.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        # Scalars
        cache_block_size,
        k_scale,
        v_scale,
        # Constexprs
        head_size,
        apply_fp8_scaling,
        is_rocm,
    )
