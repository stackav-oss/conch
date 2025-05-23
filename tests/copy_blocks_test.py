# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton copy_blocks."""

import random
from typing import Final

import pytest
import torch

from conch.ops.vllm.copy_blocks import copy_blocks as copy_blocks_triton
from conch.platforms import current_platform
from conch.reference.vllm.copy_blocks import copy_blocks as copy_blocks_reference
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.half, torch.bfloat16, torch.float, torch.uint8]
_NUM_LAYERS: Final = [4]
_NUM_HEADS: Final = [1, 4]
_HEAD_SIZES: Final = [128]
_BLOCK_SIZES: Final = [32]
_NUM_BLOCKS: Final = [1000]
_NUM_MAPPINGS: Final = [256]


@pytest.mark.parametrize("num_mappings", _NUM_MAPPINGS)
@pytest.mark.parametrize("num_layers", _NUM_LAYERS)
@pytest.mark.parametrize("num_heads", _NUM_HEADS)
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("block_size", _BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", _NUM_BLOCKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch.inference_mode()
def test_copy_blocks(
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    """Test Triton copy_blocks vs. reference implementations."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping: list[tuple[int, int]] = []
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping.append((src, dst1))
        block_mapping.append((src, dst2))

    # Create the KV caches.
    if dtype == torch.uint8:
        key_caches = [
            torch.randint(0, 255, (num_blocks, block_size * num_heads * head_size), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        value_caches = [
            torch.randint(0, 255, (num_blocks, block_size * num_heads * head_size), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
    else:
        key_caches = [
            torch.randn((num_blocks, block_size * num_heads * head_size), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        value_caches = [
            torch.randn((num_blocks, block_size * num_heads * head_size), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    # Clone the KV caches (must happen before calling either kernel bc modifications happen ieplace).
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Run the reference implementation.
    copy_blocks_reference(cloned_key_caches, cloned_value_caches, block_mapping)

    # Convert mapping list to tensor
    block_mapping_tensor = torch.tensor(block_mapping, dtype=torch.int64, device=device).view(-1, 2)

    # Call Triton kernel
    copy_blocks_triton(key_caches, value_caches, block_mapping_tensor)

    # Compare the results.
    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches, strict=False):
        torch.testing.assert_close(key_cache, cloned_key_cache)
    for value_cache, cloned_value_cache in zip(value_caches, cloned_value_caches, strict=False):
        torch.testing.assert_close(value_cache, cloned_value_cache)
