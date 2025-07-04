# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for Triton reshape_and_cache."""

import random
from typing import Final

import pytest
import torch

from conch.ops.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_conch
from conch.platforms import current_platform
from conch.reference.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_reference
from conch.third_party.vllm.utils import create_kv_cache_with_random, seed_everything

_DTYPES: Final = [torch.float16, torch.bfloat16, torch.float32]
_NUM_TOKENS: Final = [20, 40, 60]
_NUM_HEADS: Final = [1, 4, 6]
_HEAD_SIZES: Final = [64, 96, 128]
_BLOCK_SIZES: Final = [32, 128]
_NUM_BLOCKS: Final = [1000, 1500]
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
    key_cache_ref, value_cache_ref = create_kv_cache_with_random(
        num_blocks,
        block_size,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    if "fp8" in kv_cache_dtype:
        fp8_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn
        key_cache_ref = key_cache_ref.view(fp8_dtype)
        value_cache_ref = value_cache_ref.view(fp8_dtype)

    key_cache_conch = key_cache_ref.clone()
    value_cache_conch = value_cache_ref.clone()

    # Run the reference implementation.
    reshape_and_cache_reference(
        key, value, key_cache_ref, value_cache_ref, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Call Triton kernel
    reshape_and_cache_conch(
        key, value, key_cache_conch, value_cache_conch, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Can't compare FP8 directly, so bitcast to uint8
    if "fp8" in kv_cache_dtype:
        key_cache_ref = key_cache_ref.view(torch.uint8)
        value_cache_ref = value_cache_ref.view(torch.uint8)
        key_cache_conch = key_cache_conch.view(torch.uint8)
        value_cache_conch = value_cache_conch.view(torch.uint8)

    # Compare the results.
    torch.testing.assert_close(key_cache_conch, key_cache_ref)
    torch.testing.assert_close(value_cache_conch, value_cache_ref)
