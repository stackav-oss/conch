# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton reshape_and_cache."""

import random
from typing import Final

import pytest
import torch

from conch.ops.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_triton
from conch.platforms import current_platform
from conch.reference.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_reference
from conch.third_party.vllm.utils import create_kv_caches_with_random, reshape_vllm_kvcache, seed_everything

_DTYPES: Final = [torch.float16, torch.bfloat16, torch.float32]
_NUM_TOKENS: Final = [20, 40, 60]
_NUM_HEADS: Final = [1, 4]
_HEAD_SIZES: Final = [128]
_BLOCK_SIZES: Final = [32, 128]
_NUM_BLOCKS: Final = [1000]
_KV_CACHE_DTYPE: Final = ["auto", "fp8"]


@pytest.mark.parametrize("num_tokens", _NUM_TOKENS)
@pytest.mark.parametrize("num_heads", _NUM_HEADS)
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("block_size", _BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", _NUM_BLOCKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", _KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
) -> None:
    """Test Triton reshape_and_cache vs. reference implementations."""
    if kv_cache_dtype != "auto" and not current_platform.supports_fp8():
        pytest.skip("FP8 is not supported on this GPU type.")

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    kv = torch.randn((num_tokens, 2, num_heads, head_size), dtype=dtype)
    key, value = kv.unbind(dim=1)

    k_scale = torch.full((1,), 2.0, dtype=torch.float32, device=device)
    v_scale = torch.full((1,), 3.0, dtype=torch.float32, device=device)

    # Create the KV caches.
    key_caches_vllm, value_caches_vllm = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    key_cache_vllm, value_cache_vllm = key_caches_vllm[0], value_caches_vllm[0]
    key_cache_conch, value_cache_conch = reshape_vllm_kvcache(key_cache_vllm, value_cache_vllm)

    # Run the reference implementation.
    reshape_and_cache_reference(
        key, value, key_cache_vllm, value_cache_vllm, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Call Triton kernel
    reshape_and_cache_triton(
        key, value, key_cache_conch, value_cache_conch, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Reshape vLLM key/value caches
    key_cache_vllm, value_cache_vllm = reshape_vllm_kvcache(key_cache_vllm, value_cache_vllm)

    # Compare the results.
    torch.testing.assert_close(key_cache_conch, key_cache_vllm)
    torch.testing.assert_close(value_cache_conch, value_cache_vllm)
