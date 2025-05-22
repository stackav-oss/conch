# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of vLLM copy_blocks_kernel."""

from typing import Final

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _copy_blocks_kernel(
    # Pointers to tensors
    key_cache_ptrs: tl.tensor,
    value_cache_ptrs: tl.tensor,
    block_mapping_ptr: tl.const,
    # Strides of relevant tensors
    kv_cache_block_stride: int,
    block_mapping_pair_stride: int,
    # Scalars
    page_size: int,
    # Constexprs
    kv_cache_dtype: tl.constexpr,
    cxpr_chunk_size: tl.constexpr,
) -> None:
    """Implementation of copy_blocks kernel.

    Args:
        key_cache_ptrs: Tensor where each element is a pointer to a key cache block, shape: (num_layers,); cache_shape: (num_blocks, page_size).
        value_cache_ptrs: Tensor where each element is a pointer to a value cache block, shape: (num_layers,); cache_shape: (num_blocks, page_size).
        block_mapping_ptr: Tensor listing source/destination pairs to copy, shape: (num_pairs, 2).
        kv_cache_block_stride: Stride of key/value cache tensors in the 0th dimension.
        block_mapping_pair_stride: Stride of block mapping tensor in the 0th dimension.
        page_size: Size of a page (set of cache blocks).
        kv_cache_dtype: Data type of KV cache.
        cxpr_chunk_size: Size of chunk of memory to process at once.
    """
    # What layer of the cache are we processing?
    layer_index = tl.program_id(0)
    # What pair of src/dst blocks are we copying?
    pair_index = tl.program_id(1)

    # Consider adding additional program axis here to parallelize across number of chunks.
    # Could choose to loop or spawn more programs depending on num_chunks.

    # Map constexpr cache dtype to tl.dtype (cannot pass directly to kernel w/o constexpr and cannot use constexpr as argument to tl.pointer_type())
    triton_kv_cache_dtype = tl.float32
    if kv_cache_dtype == tl.float16:
        triton_kv_cache_dtype = tl.float16
    if kv_cache_dtype == tl.bfloat16:
        triton_kv_cache_dtype = tl.bfloat16
    if kv_cache_dtype == tl.uint8:
        triton_kv_cache_dtype = tl.uint8

    # Get pointers to the key/value cache tensors for this layer
    layer_key_cache_ptr = tl.load(key_cache_ptrs + layer_index).to(tl.pointer_type(triton_kv_cache_dtype))
    layer_value_cache_ptr = tl.load(value_cache_ptrs + layer_index).to(tl.pointer_type(triton_kv_cache_dtype))

    # Calculate offset to pair of src/dst block ids
    pair_offset = pair_index * block_mapping_pair_stride

    # Load the source/destination block ids
    source_block = tl.load(block_mapping_ptr + pair_offset)
    destination_block = tl.load(block_mapping_ptr + pair_offset + 1)

    # Calculate offset from the start of the k/v caches to the start of the source/destination blocks
    source_block_offset = source_block * kv_cache_block_stride
    destination_block_offset = destination_block * kv_cache_block_stride

    # We may not have enough memory to store the entire page in SRAM, so we copy in sizes of {cxpr_chunk_size}
    num_chunks = tl.cdiv(page_size, cxpr_chunk_size)
    for chunk_index in tl.range(0, num_chunks):
        # Get offsets from the start of the k/v caches for this chunk of the page
        chunk_offsets = (chunk_index * cxpr_chunk_size) + tl.arange(0, cxpr_chunk_size)
        # Handle out-of-bounds accesses
        chunk_mask = chunk_offsets < page_size

        # Load the source key block
        source_key_block = tl.load(layer_key_cache_ptr + source_block_offset + chunk_offsets, mask=chunk_mask)
        # Store it at the destination key block
        tl.store(layer_key_cache_ptr + destination_block_offset + chunk_offsets, source_key_block, mask=chunk_mask)

        # Load the source value block
        source_value_block = tl.load(layer_value_cache_ptr + source_block_offset + chunk_offsets, mask=chunk_mask)
        # Store it at the destination value block
        tl.store(layer_value_cache_ptr + destination_block_offset + chunk_offsets, source_value_block, mask=chunk_mask)


def _torch_to_triton_dtype(torch_dtype: torch.dtype) -> tl.dtype:
    """Convert Torch dtype to corresponding Triton dtype.

    Args:
        torch_dtype: Torch datatype.

    Raises:
        ValueError: if cannot match Torch dtype to corresponding Triton dtype.

    Returns:
        Corresponding Triton datatype.
    """
    if torch_dtype in (torch.float32, torch.float):
        return tl.float32

    if torch_dtype in (torch.float16, torch.half):
        return tl.float16

    if torch_dtype == torch.bfloat16:
        return tl.bfloat16

    if torch_dtype == torch.uint8:
        return tl.uint8

    msg = f"Cannot map Torch dtype ({torch_dtype}) to Triton dtype"
    raise ValueError(msg)


def copy_blocks_launcher(
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """Launch copy_blocks kernel.

    Args:
        key_caches: List of key_cache for a set of layers, len: num_layers, each element shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        value_caches: List of value_cache for a set of layers, len: num_layers, num_layers, each element shape: (num_blocks, cache_block_size, num_kv_heads, head_size).
        block_mapping: Tensor in form of list of source/destination pairs, shape: (num_pairs, 2)
    """
    # Assume sizes already checked if calling launcher. For interface with strict size checking, call `copy_blocks()`.
    num_layers: Final = len(key_caches)
    num_pairs: Final = block_mapping.size(0)
    device: Final = key_caches[0].device
    kv_cache_dtype: Final = key_caches[0].dtype
    kv_cache_block_stride: Final = key_caches[0].stride(0)
    block_mapping_pair_stride: Final = block_mapping.stride(0)
    page_size: Final = key_caches[0].size(1)

    key_cache_ptrs = [key_cache_tensor.data_ptr() for key_cache_tensor in key_caches]
    key_cache_ptrs_tensor = torch.tensor(key_cache_ptrs, device=device, dtype=torch.uint64)

    value_cache_ptrs = [value_cache_tensor.data_ptr() for value_cache_tensor in value_caches]
    value_cache_ptrs_tensor = torch.tensor(value_cache_ptrs, device=device, dtype=torch.uint64)

    triton_kv_cache_dtype: Final = _torch_to_triton_dtype(kv_cache_dtype)

    cxpr_chunk_size: tl.constexpr = 1024

    # Parallelize over each layer and each pair of copies
    grid = (num_layers, num_pairs)

    # Launch kernel
    _copy_blocks_kernel[grid](
        key_cache_ptrs_tensor,
        value_cache_ptrs_tensor,
        block_mapping,
        kv_cache_block_stride,
        block_mapping_pair_stride,
        page_size,
        triton_kv_cache_dtype,
        cxpr_chunk_size,
    )
