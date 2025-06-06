# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test Triton PagedAttention."""

import math
from typing import Final

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from einops import einsum, rearrange

from conch import envs
from conch.ops.attention.paged_attention import paged_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything

_ENABLE_VLLM: Final = envs.CONCH_ENABLE_VLLM and current_platform.has_cuda()
_BATCH_SIZES: Final = [1, 4, 8]
_CACHE_BLOCK_SIZES_VLLM: Final = [16, 32]
_CACHE_BLOCK_SIZES_FLASH: Final = [16, 32, 64]
_HEAD_SIZES_VLLM: Final = [64, 80, 96, 112, 128, 192, 256]
_HEAD_SIZES_FLASH: Final = [64, 96, 128, 256]
_SEQUENCE_LENGTHS: Final = [16, 128, 240, 333, 1002, 2048]
# MHA, MQA, and GQA
# - MHA: num_query_heads == num_kv_heads
# - MQA: num_kv_heads == 1
# - GQA: num_query_heads != num_kv_heads && num_kv_heads != 1
_NUM_HEADS: Final = [(8, 8), (4, 4), (8, 1), (4, 1), (16, 4), (16, 2), (8, 4), (8, 2)]
# Too many parameterizations makes the SDPA test cases too slow
_BATCH_SIZES_ABRIDGED: Final = [4]
_SEQUENCE_LENGTHS_ABRIDGED: Final = [240, 333, 1002]
_NUM_HEADS_ABRIDGED: Final = [(8, 8), (4, 1), (16, 4)]


def _get_tolerance_for_dtype(dtype: torch.dtype) -> float:
    """Get expected tolerance to match to for a given dtype."""
    if dtype == torch.float16:
        return 5e-3

    if dtype == torch.bfloat16:
        return 3e-2

    if dtype == torch.float32:
        return 2e-3

    msg = f"Unsupported dtype: '{dtype}'"
    raise NotImplementedError(msg)


def _create_paged_cache_from_contiguous_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    num_cache_blocks: int,
    cache_block_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paged cache from contiguous KV.

    Args:
        key: Key tensor.
        value: Value tensor.
        num_cache_blocks: Number of cache blocks in the kv caches.
        cache_block_size: Size of each cache block.
        dtype: Data type of tensors.

    Returns:
        Tuple of paged key cache, paged value cache, and block tables for traversing caches.
    """
    assert key.shape == value.shape
    assert key.device == value.device

    device: Final = key.device

    batch_size, num_kv_heads, sequence_length, head_size = key.shape
    max_num_pages_per_seq: Final = num_cache_blocks // batch_size
    max_seq_len = max_num_pages_per_seq * cache_block_size
    assert sequence_length <= max_seq_len

    k_cache = torch.empty(num_cache_blocks, cache_block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    v_cache = torch.empty(num_cache_blocks, cache_block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    block_table = torch.zeros((batch_size, max_num_pages_per_seq), dtype=torch.int32, device=device)

    current_page = 0
    current_page_offset = 0

    for batch_index in range(batch_size):
        current_blocks = []
        for seq_index in range(sequence_length):
            if current_page not in current_blocks:
                current_blocks.append(current_page)

            for head_index in range(num_kv_heads):
                current_k = key[batch_index][head_index][seq_index].detach().clone()
                current_v = value[batch_index][head_index][seq_index].detach().clone()

                k_cache[current_page][current_page_offset][head_index] = current_k
                v_cache[current_page][current_page_offset][head_index] = current_v

            current_page_offset += 1

            if current_page_offset == cache_block_size:
                current_page += 1
                current_page_offset = 0

        block_table[batch_index] = torch.Tensor(current_blocks)

        # Next batch should get its own page, so increment page index and reset offset
        if current_page_offset != 0:
            current_page += 1
            current_page_offset = 0

    return k_cache, v_cache, block_table


def _convert_paged_to_contiguous(
    k_cache_paged: torch.Tensor,
    v_cache_paged: torch.Tensor,
    block_table: torch.Tensor,
    num_kv_heads: int,
    seq_len: int,
    head_size: int,
    cache_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert paged kv cache to contiguous.

    Args:
        k_cache_paged: Paged key cache.
        v_cache_paged: Paged value cache.
        block_table: Block tables for traversing kv caches.
        num_kv_heads: The number of kv heads.
        seq_len: The max sequence length.
        head_size: The dimension of the attention head.
        cache_block_size: The size of each cache block.

    Returns:
        Tuple of key cache in contiguous memory and value cache in contiguous memory.
    """
    batch_size, _ = block_table.shape

    k_contiguous = torch.empty(
        (batch_size, seq_len, num_kv_heads, head_size), dtype=k_cache_paged.dtype, device=k_cache_paged.device
    )
    v_contiguous = torch.empty(
        (batch_size, seq_len, num_kv_heads, head_size), dtype=v_cache_paged.dtype, device=v_cache_paged.device
    )

    for batch_index in range(batch_size):
        current_seq_len_begin = 0
        current_seq_len_end = current_seq_len_begin + cache_block_size

        this_k = k_contiguous[batch_index]
        this_v = v_contiguous[batch_index]

        these_blocks = block_table[batch_index]

        for relative_block_index, cache_block_index in enumerate(these_blocks):
            current_k_block = k_cache_paged[cache_block_index]
            current_v_block = v_cache_paged[cache_block_index]

            # The end of the current cache block or the end of the sequence, whichever is smaller
            actual_end = min(current_seq_len_end, seq_len)

            # Relative end is the end index in the current cache block
            relative_end = actual_end - (relative_block_index * cache_block_size)

            this_k[current_seq_len_begin:actual_end, :, :] = current_k_block.view(
                cache_block_size,
                num_kv_heads,
                head_size,
            )[:relative_end, :, :]

            this_v[current_seq_len_begin:actual_end, :, :] = current_v_block.view(
                cache_block_size,
                num_kv_heads,
                head_size,
            )[:relative_end, :, :]

            current_seq_len_begin += cache_block_size
            current_seq_len_end += cache_block_size

    return k_contiguous, v_contiguous


def _gqa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Grouped Query Attention."""
    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape

    if not (bq == bk == bv and dq == dk == dv):
        msg = f"Expected query, key, and value to have the same batch size (dim=0) and embedding dimension (dim=3), but got query: {query.shape}, key: {key.shape}, and value: {value.shape}."
        raise ValueError(msg)

    if (hk != hv) or (nk != nv):
        msg = f"Expected key and value to have the same size in dimensions 1 and 2, but got key: {key.shape} and value: {value.shape}."
        raise ValueError(msg)

    if hq % hk != 0:
        msg = f"Expected query heads to be a multiple of key/value heads, but got query: {query.shape} and key/value: {key.shape}."
        raise ValueError(msg)

    num_head_groups = hq // hk

    query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
    similarity = einsum(query, key, "b g h n d, b h s d -> b g h n s")
    attention = F.softmax(similarity * scale, dim=-1)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
    # Move head dimension back to axis 2
    return rearrange(out, "b g h n d -> b (h g) n d")


def _run_paged_vs_sdpa(
    batch_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    cache_block_size: int,
    dtype: torch.dtype,
    gqa_workaround: bool,
) -> None:
    """Run test case comparing our Triton PagedAttention kernel vs PyTorch Scaled Dot Product Attention.

    Args:
        batch_size: Number of batches in input.
        head_size: Head dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        sequence_length: Sequence length of input.
        cache_block_size: Number of K/V tensors that can fit in a cache block.
        dtype: Data type of tensors.
        gqa_workaround: Enable workaround for Grouped Query Attention. PyTorch SDPA does not currently
            support GQA so use alternative ground-truth function.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    # There are some small discrepancies in our output that I believe are caused by fp32<->fp16 conversions inside of the Triton kernel.
    atol: Final = _get_tolerance_for_dtype(dtype)
    rtol: Final = 1e-3

    scale: Final = float(1.0 / (head_size**0.5))

    cache_blocks_per_seq: Final = math.ceil(sequence_length / cache_block_size)
    total_cache_size: Final = batch_size * cache_blocks_per_seq * cache_block_size
    num_cache_blocks: Final = total_cache_size // cache_block_size

    q = torch.randn(
        batch_size, num_query_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )

    sequence_lengths = torch.full((batch_size,), sequence_length, dtype=torch.int32, device=device)

    key_cache_paged, value_cache_paged, block_table = _create_paged_cache_from_contiguous_kv(
        k, v, num_cache_blocks, cache_block_size, dtype
    )
    kc_duplicate, vc_duplicate = _convert_paged_to_contiguous(
        key_cache_paged,
        value_cache_paged,
        block_table,
        num_kv_heads,
        sequence_length,
        head_size,
        cache_block_size,
    )

    kc_duplicate = kc_duplicate.permute(0, 2, 1, 3)
    vc_duplicate = vc_duplicate.permute(0, 2, 1, 3)

    torch.testing.assert_close(k, kc_duplicate, atol=atol, rtol=rtol)
    torch.testing.assert_close(v, vc_duplicate, atol=atol, rtol=rtol)

    q_sdpa = q.detach().clone()
    k_sdpa = k.detach().clone()
    v_sdpa = v.detach().clone()

    if gqa_workaround:
        # Hand-rolled GQA impl because PyTorch does not support it via scaled_dot_product_attention
        out_sdpa = _gqa(q_sdpa, k_sdpa, v_sdpa, scale)
    else:
        # Default PyTorch SDPA
        out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=scale)

    final_out_sdpa = out_sdpa[:, :, -1, :]

    q_paged = q[:, :, -1, :]

    out_paged = paged_attention(
        q_paged,
        key_cache_paged,
        value_cache_paged,
        block_table,
        sequence_lengths,
        scale=scale,
    )

    torch.testing.assert_close(final_out_sdpa, out_paged, atol=atol, rtol=rtol)


def _triton_vs_vllm_cuda(
    batch_size: int,
    cache_block_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    kv_cache_dtype: str,
    dtype: torch.dtype,
) -> None:
    """Run test case comparing our Triton PagedAttention kernel vs vLLM CUDA PagedAttention.

    Args:
        batch_size: Number of batches in input.
        head_size: Head dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        sequence_length: Sequence length of input.
        cache_block_size: Number of K/V tensors that can fit in a cache block.
        kv_cache_dtype: Data type of KV cache.
        dtype: Datatype for tensors.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    from vllm._custom_ops import paged_attention_v2 as vllm_paged_attention_v2

    query, key_cache_vllm, value_cache_vllm, key_cache_conch, value_cache_conch, block_table, seq_lens = create_tensors(
        head_size,
        sequence_length,
        cache_block_size,
        batch_size,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        "cuda",
        dtype,
    )

    scale: Final = float(1.0 / (head_size**0.5))
    atol: Final = 1e-3
    rtol: Final = 1e-3

    k_scale = torch.full((1,), 0.5)
    v_scale = torch.full((1,), 0.5)

    # Run vLLM reference implementation
    output_vllm = torch.empty_like(query)

    max_seq_len = torch.max(seq_lens)

    partition_size = 512
    max_num_partitions = (max_seq_len + partition_size - 1) // partition_size
    tmp_output = torch.empty(
        size=(batch_size, num_query_heads, max_num_partitions, head_size),
        dtype=dtype,
        device=query.device,
    )
    exp_sums = torch.empty(
        size=(batch_size, num_query_heads, max_num_partitions),
        dtype=torch.float32,
        device=query.device,
    )
    max_logits = torch.empty_like(exp_sums)

    vllm_paged_attention_v2(
        output_vllm,
        max_logits,
        exp_sums,
        tmp_output,
        query,
        key_cache_vllm,
        value_cache_vllm,
        num_kv_heads,
        scale,
        block_table,
        seq_lens,
        cache_block_size,
        int(max_seq_len.item()),
        None,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )

    # Run Triton implementation
    output_conch = paged_attention(
        query,
        key_cache_conch,
        value_cache_conch,
        block_table,
        seq_lens,
        scale=scale,
        softcap=0.0,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(output_vllm, output_conch, atol=atol, rtol=rtol)


def _triton_vs_flash_attn(
    batch_size: int,
    cache_block_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    dtype: torch.dtype,
    apply_softcap: bool,
) -> None:
    """Run test case comparing our Triton PagedAttention kernel vs CUDA FlashAttnWithKVCache.

    Args:
        batch_size: Number of batches in input.
        head_size: Head dimension.
        num_query_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        sequence_length: Sequence length of input.
        cache_block_size: Number of K/V tensors that can fit in a cache block.
        dtype: Datatype for tensors.
        apply_softcap: Whether or not to apply softcapping.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    kv_cache_dtype = "auto"

    from vllm.vllm_flash_attn import flash_attn_with_kvcache  # type: ignore[attr-defined, unused-ignore]

    query, _, _, key_cache, value_cache, block_table, seq_lens = create_tensors(
        head_size,
        sequence_length,
        cache_block_size,
        batch_size,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        "cuda",
        dtype,
    )

    softcap: float = 0.0
    if apply_softcap:
        # Gemma2 default for self-attention layers
        softcap = 30.0
        # Ensure the values of qk are at least within softcap range.
        query = torch.randn_like(query) * softcap

    scale: Final = float(1.0 / (head_size**0.5))

    # There are some small discrepancies in our output that I believe are caused by fp32<->fp16 conversions inside of the Triton kernel.
    atol: Final = 1e-3 if not apply_softcap else 2e-2
    rtol: Final = 1e-3

    # Run FlashAttnWithKVCache implementation
    query_fa = query.unsqueeze(1)

    output_fa = flash_attn_with_kvcache(
        query_fa,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=seq_lens,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        softcap=softcap,
    )

    output_fa = output_fa.squeeze(1)

    # Run Triton implementation
    output_conch = paged_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        scale=scale,
        softcap=softcap,
        kv_cache_dtype=kv_cache_dtype,
    )

    assert torch.allclose(output_fa, output_conch, atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch_size", _BATCH_SIZES_ABRIDGED)
@pytest.mark.parametrize("cache_block_size", _CACHE_BLOCK_SIZES_VLLM)
@pytest.mark.parametrize("head_size", _HEAD_SIZES_VLLM)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS_ABRIDGED)
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS_ABRIDGED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_paged_attention(
    batch_size: int,
    cache_block_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    dtype: torch.dtype,
) -> None:
    """Test PagedAttention Triton kernel with various configurations vs. PyTorch."""
    gqa_workaround = num_query_heads // num_kv_heads not in (1, num_query_heads)

    _run_paged_vs_sdpa(
        batch_size,
        head_size,
        num_query_heads,
        num_kv_heads,
        sequence_length,
        cache_block_size,
        dtype,
        gqa_workaround,
    )


@pytest.mark.skipif(not _ENABLE_VLLM, reason="This test case requires vLLM")
@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
@pytest.mark.parametrize("cache_block_size", _CACHE_BLOCK_SIZES_VLLM)
@pytest.mark.parametrize("head_size", _HEAD_SIZES_VLLM)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS)
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_vs_vllm_cuda(
    batch_size: int,
    cache_block_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    kv_cache_dtype: str,
    dtype: torch.dtype,
) -> None:
    """Test PagedAttention kernel with various configurations vs. vLLM CUDA PagedAttention."""
    if sequence_length < cache_block_size:
        pytest.skip()

    if kv_cache_dtype == "fp8" and not current_platform.supports_fp8():
        pytest.skip()

    _triton_vs_vllm_cuda(
        batch_size,
        cache_block_size,
        head_size,
        num_query_heads,
        num_kv_heads,
        sequence_length,
        kv_cache_dtype,
        dtype,
    )


@pytest.mark.skipif(not _ENABLE_VLLM, reason="This test case requires vLLM")
@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
@pytest.mark.parametrize("cache_block_size", _CACHE_BLOCK_SIZES_FLASH)
@pytest.mark.parametrize("head_size", _HEAD_SIZES_FLASH)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS)
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS)
@pytest.mark.parametrize("apply_softcap", [True, False])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_vs_flash_attn(
    batch_size: int,
    cache_block_size: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    sequence_length: int,
    kv_cache_dtype: str,
    apply_softcap: bool,
    dtype: torch.dtype,
) -> None:
    """Test PagedAttention kernel with various configurations vs. CUDA FlashAttnWithKVCache."""
    # Only supported on CUDA
    if current_platform.is_amd():
        pytest.skip()

    if sequence_length < cache_block_size:
        pytest.skip()

    _triton_vs_flash_attn(
        batch_size,
        cache_block_size,
        head_size,
        num_query_heads,
        num_kv_heads,
        sequence_length,
        dtype,
        apply_softcap,
    )
