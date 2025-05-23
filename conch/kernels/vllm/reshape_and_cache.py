# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

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
    k_scale_ptr: tl.tensor,
    v_scale_ptr: tl.tensor,
    # Strides of relevant tensors
    k_token_stride: int,
    k_head_stride: int,
    k_head_element_stride: int,
    v_token_stride: int,
    v_head_stride: int,
    v_head_element_stride: int,
    kv_cache_page_stride: int,
    kv_cache_block_stride: int,
    kv_cache_head_stride: int,
    kv_cache_head_element_stride: int,
    # Scalars
    cache_block_size: int,
    # Constexprs
    cxpr_head_size: tl.constexpr,
    cxpr_apply_fp8_scaling: tl.constexpr,
    cxpr_is_rocm: tl.constexpr,
) -> None:
    """Implementation of reshape_and_cache kernel.

    Args:
        key_ptr: Pointer to tensor of new key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value_ptr: Pointer to tensor of new value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache_ptr: Pointer to tensor of key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache_ptr: Pointer to tensor of value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping_ptr: Pointer to slot mapping tensor, shape: (num_tokens,).
        k_scale_ptr: Pointer to Fp8 scaling factor for k.
        v_scale_ptr: Pointer to Fp8 scaling factor for v.
        k_token_stride: Stride of key tensor in 0th dimension.
        k_head_stride: Stride of key tensor in 1st dimension.
        k_head_element_stride: Stride of key tensor in 2nd dimension.
        v_token_stride: Stride of value tensor in 0th dimension.
        v_head_stride: Stride of value tensor in 1st dimension.
        v_head_element_stride: Stride of value tensor in 2nd dimension.
        kv_cache_page_stride: Stride of key/value cache tensors in 0th dimension.
        kv_cache_block_stride: Stride of key/value cache tensors in 1st dimension.
        kv_cache_head_stride: Stride of key/value cache tensors in 2nd dimension.
        kv_cache_head_element_stride: Stride of key/value cache tensors in 3rd dimension.
        cache_block_size: Size of each cache block / page in the KV cache.
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

    # Calculate index of page (value in range(0, num_pages))
    page_index = slot_index // cache_block_size
    # Calculate entry index inside of a cache block/page for this slot (value in range(0, cache_block_size))
    entry_index = slot_index % cache_block_size

    # Calculate offset into key/value tensors to get to the token for this program
    k_token_offset = token_index * k_token_stride
    v_token_offset = token_index * v_token_stride
    # Calculate offset into key/value tensors to get to the head for this program
    k_head_offset = head_index * k_head_stride
    v_head_offset = head_index * v_head_stride
    # Offsets for each element of the head
    k_head_offsets = tl.arange(0, cxpr_head_size) * k_head_element_stride
    v_head_offsets = tl.arange(0, cxpr_head_size) * v_head_element_stride

    # Load key/value vectors for this token/head
    key = tl.load(key_ptr + k_token_offset + k_head_offset + k_head_offsets)
    value = tl.load(value_ptr + v_token_offset + v_head_offset + v_head_offsets)

    # Apply FP8 scaling if necessary
    if cxpr_apply_fp8_scaling:
        k_scale = tl.load(k_scale_ptr)
        k_scale = 1.0 / k_scale
        key *= k_scale

        v_scale = tl.load(v_scale_ptr)
        v_scale = 1.0 / v_scale
        value *= v_scale

        fp8_dtype = tl.float8e4b8 if cxpr_is_rocm else tl.float8e4nv

        # First, cast to the lower-precision floating point type, then bitcast the fp8 representation
        # to uint8 to match the dtype of the cache
        key = key.to(fp8_dtype).to(key_cache_ptr.dtype.element_ty, bitcast=True)
        value = value.to(fp8_dtype).to(value_cache_ptr.dtype.element_ty, bitcast=True)

    # Calculate offset into key/value cache tensors to get to the cache block we're copying into
    kv_page_offset = page_index * kv_cache_page_stride
    # Calculate offset in a cache block to get to the entry for we're copying into
    kv_cache_entry_offset = entry_index * kv_cache_block_stride
    # Calculate offset into key/value cache tensors to get to the head we're copying into
    kv_cache_head_offset = head_index * kv_cache_head_stride

    # Offsets for each element of the head
    kv_head_offsets = tl.arange(0, cxpr_head_size) * kv_cache_head_element_stride

    # Store key/value vectors into cache
    tl.store(key_cache_ptr + kv_page_offset + kv_cache_entry_offset + kv_cache_head_offset + kv_head_offsets, key)
    tl.store(value_cache_ptr + kv_page_offset + kv_cache_entry_offset + kv_cache_head_offset + kv_head_offsets, value)


def reshape_and_cache_launcher(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Launch reshape_and_cache kernel.

    Args:
        key: New key vectors, shape: (num_tokens, num_kv_heads, head_size).
        value: New value vectors, shape: (num_tokens, num_kv_heads, head_size).
        key_cache: Key cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        value_cache: Value cache, shape: (num_pages, cache_block_size, num_kv_heads, head_size).
        slot_mapping: Tensor listing what slots in the cache each key/value vector should be placed in, shape: (num_tokens,).
        kv_cache_dtype: String datatype of kv cache elements.
        k_scale: Fp8 scaling factor for k.
        v_scale: Fp8 scaling factor for v.
    """
    # Assume sizes already checked if calling launcher. For interface with strict size checking, call `reshape_and_cache()`.
    _, num_kv_heads, head_size = key.shape
    num_pages, cache_block_size, _, _ = key_cache.shape

    # Note: In vLLM v1, slot_mapping is the only tensor that can be trusted to tell the correct number of tokens
    num_tokens = slot_mapping.size(0)

    assert key.shape == value.shape  # noqa: S101
    assert key_cache.shape == value_cache.shape  # noqa: S101

    assert key_cache.stride(0) == value_cache.stride(0)  # noqa: S101
    assert key_cache.stride(1) == value_cache.stride(1)  # noqa: S101
    assert key_cache.stride(2) == value_cache.stride(2)  # noqa: S101
    assert key_cache.stride(3) == value_cache.stride(3)  # noqa: S101
    assert key_cache.stride(3) == 1  # noqa: S101

    assert cache_block_size == triton.next_power_of_2(cache_block_size), "Cache block size must be a power of two!"  # noqa: S101
    assert head_size == triton.next_power_of_2(head_size), "Head size must be a power of two!"  # noqa: S101

    is_rocm: tl.constexpr = current_platform.is_amd()
    apply_fp8_scaling: tl.constexpr = kv_cache_dtype == "fp8" or kv_cache_dtype == "fp8_e4m3"

    if apply_fp8_scaling:
        assert k_scale is not None  # noqa: S101
        assert v_scale is not None  # noqa: S101
        assert k_scale.numel() == 1  # noqa: S101
        assert v_scale.numel() == 1  # noqa: S101

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
        k_scale,
        v_scale,
        # Strides of relevant tensors
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        # Scalars
        cache_block_size,
        # Constexprs
        head_size,
        apply_fp8_scaling,
        is_rocm,
    )
