# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch reference implementation of vLLM copy_blocks."""

import torch

from conch import envs
from conch.platforms import current_platform


def _copy_blocks_pytorch_ref(
    key_caches: list[torch.Tensor], value_caches: list[torch.Tensor], block_mapping: list[tuple[int, int]]
) -> None:
    """Reference PyTorch-only implementation of copy_blocks."""
    for src, dst in block_mapping:
        for key_cache in key_caches:
            key_cache[dst].copy_(key_cache[src])
        for value_cache in value_caches:
            value_cache[dst].copy_(value_cache[src])


def _copy_blocks_vllm_ref(
    key_caches: list[torch.Tensor], value_caches: list[torch.Tensor], block_mapping: list[tuple[int, int]]
) -> None:
    """Reference vLLM implementation of copy_blocks."""
    from vllm._custom_ops import copy_blocks as copy_blocks_vllm

    block_mapping_tensor = torch.tensor(block_mapping, dtype=torch.int64, device=key_caches[0].device).view(-1, 2)
    copy_blocks_vllm(key_caches, value_caches, block_mapping_tensor)


def copy_blocks(
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    block_mapping: list[tuple[int, int]],
) -> None:
    """Reference implementation of vLLM's copy_blocks operation."""
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        return _copy_blocks_vllm_ref(key_caches, value_caches, block_mapping)

    return _copy_blocks_pytorch_ref(key_caches, value_caches, block_mapping)
