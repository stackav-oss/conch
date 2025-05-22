# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test for rotary_embedding."""

from typing import Final

import pytest
import torch

from conch.ops.embedding.rotary_embedding import rotary_embedding as rotary_embedding_triton
from conch.platforms import current_platform
from conch.reference.embedding.rotary_embedding import compute_cos_sin_cache
from conch.reference.embedding.rotary_embedding import rotary_embedding as rotary_embedding_reference
from conch.third_party.vllm.utils import seed_everything

_SEQ_LENS: Final = [2048, 8192]
_NUM_HEADS: Final = [4, 17]
_HEAD_SIZES: Final = [128, 256]
_BASES: Final = [10000]


@pytest.mark.parametrize("seq_len", _SEQ_LENS)
@pytest.mark.parametrize("num_heads", _NUM_HEADS)
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("base", _BASES)
@torch.inference_mode()
def test_kernel(
    seq_len: int,
    num_heads: int,
    head_size: int,
    base: int,
) -> None:
    rotary_dim = head_size
    max_position = seq_len
    is_neox_style = True

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    cos_sin_cache = compute_cos_sin_cache(base, rotary_dim, max_position)

    # seq_len == num_tokens in this case
    positions = torch.randint(0, max_position, (seq_len,))
    query = torch.randn(seq_len, num_heads * head_size)
    key = torch.randn_like(query)
    query_ref = torch.clone(query)  # Need to clone since updates are performed in-place
    key_ref = torch.clone(key)

    query_ref, key_ref = rotary_embedding_reference(
        positions,
        query_ref,
        key_ref,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style=is_neox_style,
    )

    query_triton, key_triton = rotary_embedding_triton(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
    )

    torch.testing.assert_close(query_ref, query)
    torch.testing.assert_close(key_ref, key)
