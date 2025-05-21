# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Test Triton varlen attention."""

from typing import Final

import pytest
import torch

from conch import envs
from conch.ops.attention.varlen_attention import varlen_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything

_ENABLE_VLLM: Final = envs.CONCH_ENABLE_VLLM and current_platform.has_cuda()
_HEAD_SIZES: Final = [64, 96, 128, 256]
_NUM_SEQS_ABRIDGED: Final = [4, 10]
# MHA, MQA, and GQA
# - MHA: num_query_heads == num_kv_heads
# - MQA: num_kv_heads == 1
# - GQA: num_query_heads != num_kv_heads && num_kv_heads != 1
_NUM_HEADS_ABRIDGED: Final = [(8, 8), (4, 1), (16, 4)]
_SEQUENCE_LENGTHS: Final = [256, 257, 343, 1024, 1025]


def _get_tolerance_for_dtype(dtype: torch.dtype) -> float:
    """Get expected tolerance to match to for a given dtype."""
    if dtype == torch.float16:
        return 5e-4

    if dtype == torch.bfloat16:
        return 1e-3

    msg = f"Unsupported dtype: '{dtype}'"
    raise NotImplementedError(msg)


def _convert_paged_to_contiguous(
    k_cache_paged: torch.Tensor,
    v_cache_paged: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    total_num_k: int,
    num_kv_heads: int,
    head_size: int,
    cache_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert paged kv cache to contiguous.

    Args:
        k_cache_paged: Paged key cache.
        v_cache_paged: Paged value cache.
        block_tables: Block tables for traversing kv caches.
        cu_seqlens_k: Sequence lengths.
        num_kv_heads: The number of kv heads.
        total_num_k: The total number of keys/values.
        head_size: The dimension of the attention head.
        cache_block_size: The size of each cache block.

    Returns:
        Tuple of key cache in contiguous memory and value cache in contiguous memory.
    """
    batch_size, _ = block_tables.shape

    k_contiguous = torch.empty(
        (total_num_k, num_kv_heads, head_size), dtype=k_cache_paged.dtype, device=k_cache_paged.device
    )
    v_contiguous = torch.empty(
        (total_num_k, num_kv_heads, head_size), dtype=v_cache_paged.dtype, device=v_cache_paged.device
    )

    for batch_index in range(batch_size):
        current_seq_len_begin = int(cu_seqlens_k[batch_index].item())
        current_seq_len_end = int(cu_seqlens_k[batch_index + 1].item())
        this_seqlen = current_seq_len_end - current_seq_len_begin

        this_k = k_contiguous[current_seq_len_begin:current_seq_len_end]
        this_v = v_contiguous[current_seq_len_begin:current_seq_len_end]

        these_blocks = block_tables[batch_index]

        relative_begin = 0

        for relative_block_index, cache_block_index in enumerate(these_blocks):
            current_k_block = k_cache_paged[cache_block_index]
            current_v_block = v_cache_paged[cache_block_index]

            relative_seqlen = relative_block_index * cache_block_size
            if relative_seqlen >= this_seqlen:
                break

            # Relative end is the end index in the current cache block
            relative_end = relative_begin + min(this_seqlen - relative_seqlen, cache_block_size)

            this_length = relative_end - relative_begin

            this_k_block = current_k_block.permute(1, 0, 2)
            this_v_block = current_v_block.permute(1, 0, 2)

            this_k[relative_begin:relative_end, :, :] = this_k_block[:this_length, :, :]
            this_v[relative_begin:relative_end, :, :] = this_v_block[:this_length, :, :]

            relative_begin += cache_block_size

    return k_contiguous, v_contiguous


def _pytorch_attention_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention_scores = torch.matmul(q, k.transpose(-2, -1))

    attention_scaled_scores = scale * attention_scores

    if causal:
        q_len, k_len = q.shape[1], k.shape[1]
        row_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
        col_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
        col_offset = q_len - k_len
        causal_mask = row_idx >= (col_offset + col_idx)
        attention_scaled_scores = attention_scaled_scores.masked_fill(
            torch.logical_not(causal_mask.unsqueeze(0)), float("-inf")
        )

    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if causal:
        max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)

    attention_shifted_scaled_scores = attention_scaled_scores - max_scores

    exp_scores = torch.exp(attention_shifted_scaled_scores)

    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if causal:
        sum_exp_scores = torch.where(sum_exp_scores == 0, torch.ones_like(sum_exp_scores), sum_exp_scores)

    p = exp_scores / sum_exp_scores

    softmax_lse = max_scores + torch.log(sum_exp_scores)
    softmax_lse = softmax_lse.squeeze(-1)

    output = torch.matmul(p, v)

    return output, softmax_lse


def _varlen_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    causal: bool,
) -> torch.Tensor:
    batch_size = cu_seqlens_q.shape[0] - 1
    num_query_heads, num_kv_heads = q.shape[1], k.shape[1]
    head_size = q.shape[2]

    total_num_q = q.shape[0]

    output = torch.zeros((total_num_q, num_query_heads, head_size), dtype=q.dtype, device=q.device)

    group_size = num_query_heads // num_kv_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")

    for batch_index in range(batch_size):
        start_q = int(cu_seqlens_q[batch_index].item())
        end_q = int(cu_seqlens_q[batch_index + 1].item())
        start_k = int(cu_seqlens_k[batch_index].item())
        end_k = int(cu_seqlens_k[batch_index + 1].item())

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k

        this_q = q[start_q:end_q, :, :]
        this_k = k[start_k:end_k, :, :]
        this_v = v[start_k:end_k, :, :]

        this_q = this_q.permute(1, 0, 2)
        this_k = this_k.permute(1, 0, 2)
        this_v = this_v.permute(1, 0, 2)

        if group_size != 1:
            this_q = this_q.reshape(num_kv_heads, group_size, seqlen_q, head_size)
            this_k = this_k.unsqueeze(1).expand(-1, group_size, -1, -1)
            this_v = this_v.unsqueeze(1).expand(-1, group_size, -1, -1)

            this_q = this_q.reshape(num_kv_heads * group_size, seqlen_q, head_size)
            this_k = this_k.reshape(num_kv_heads * group_size, seqlen_k, head_size)
            this_v = this_v.reshape(num_kv_heads * group_size, seqlen_k, head_size)
        else:
            this_q = this_q.reshape(num_query_heads, seqlen_q, head_size)
            this_k = this_k.reshape(num_kv_heads, seqlen_k, head_size)
            this_v = this_v.reshape(num_kv_heads, seqlen_k, head_size)

        this_output, _ = _pytorch_attention_inner(this_q, this_k, this_v, scale, causal)

        if group_size != 1:
            this_output = this_output.reshape(num_kv_heads, group_size, seqlen_q, head_size)
            this_output = this_output.reshape(num_query_heads, seqlen_q, head_size)

        this_output = this_output.permute(1, 0, 2)

        output[start_q:end_q, :, :] = this_output

    return output


@pytest.mark.parametrize("num_seqs", _NUM_SEQS_ABRIDGED)
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS_ABRIDGED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS)
@pytest.mark.parametrize("causal", [True, False])
def test_varlen_attention_vs_pytorch(
    num_seqs: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    sequence_length: int,
    causal: bool,
) -> None:
    """Test Varlen Attention Triton kernel with various configurations vs. vLLM FlashAttnVarlen."""
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    scale: Final = float(1.0 / (head_size**0.5))
    softcap: Final = 0.0

    tolerance: Final = _get_tolerance_for_dtype(dtype)

    kv_cache_dtype: Final = "auto"

    cache_block_size = 16

    _, _, _, key_cache_conch, value_cache_conch, block_tables, seq_lens = create_tensors(
        head_size,
        sequence_length,
        cache_block_size,
        num_seqs,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        current_platform.device,
        dtype,
    )

    starting_item = torch.as_tensor([0], dtype=torch.int32)

    cu_seqlens_q = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    cu_seqlens_q = torch.cat((starting_item, cu_seqlens_q), dim=0)

    cu_seqlens_k = cu_seqlens_q.clone()

    max_seqlen_q = int(torch.max(seq_lens).item())
    max_seqlen_k = int(max_seqlen_q)

    total_num_q = int(cu_seqlens_q[-1].item())
    total_num_k = int(cu_seqlens_k[-1].item())

    key_contiguous, value_contiguous = _convert_paged_to_contiguous(
        key_cache_conch,
        value_cache_conch,
        block_tables,
        cu_seqlens_k,
        total_num_k,
        num_kv_heads,
        head_size,
        cache_block_size,
    )

    q = torch.empty(total_num_q, num_query_heads, head_size, dtype=dtype, device=device)
    q.uniform_(-scale, scale)

    pytorch_output = _varlen_attention_pytorch(
        q=q,
        k=key_contiguous,
        v=value_contiguous,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        causal=causal,
    )

    conch_output = varlen_attention(
        query=q,
        key_cache=key_cache_conch,
        value_cache=value_cache_conch,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        scale=scale,
        softcap=softcap,
    )

    torch.testing.assert_close(pytorch_output, conch_output, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not _ENABLE_VLLM, reason="This test case requires vLLM")
@pytest.mark.parametrize("num_seqs", _NUM_SEQS_ABRIDGED)
@pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS_ABRIDGED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("is_pure_decode", [True, False])
def test_varlen_attention_vs_flash_attn(
    num_seqs: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    sequence_length: int,
    causal: bool,
    is_pure_decode: bool,
) -> None:
    """Test Varlen Attention Triton kernel with various configurations vs. vLLM FlashAttnVarlen."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # type: ignore[attr-defined, unused-ignore]

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    scale: Final = float(1.0 / (head_size**0.5))
    softcap: Final = 0.0

    tolerance: Final = _get_tolerance_for_dtype(dtype)

    kv_cache_dtype: Final = "auto"

    cache_block_size = 16

    _, _, _, key_cache_conch, value_cache_conch, block_tables, seq_lens = create_tensors(
        head_size,
        sequence_length,
        cache_block_size,
        num_seqs,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        current_platform.device,
        dtype,
    )

    starting_item = torch.as_tensor([0], dtype=torch.int32)

    if is_pure_decode:
        seqlens_q = torch.ones((num_seqs,), dtype=torch.int32)

        cu_seqlens_q = torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.cat((starting_item, cu_seqlens_q), dim=0)

        cu_seqlens_k = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        cu_seqlens_k = torch.cat((starting_item, cu_seqlens_k), dim=0)

        max_seqlen_q = int(torch.max(seqlens_q).item())
        max_seqlen_k = int(torch.max(seq_lens).item())
    else:
        cu_seqlens_q = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.cat((starting_item, cu_seqlens_q), dim=0)

        cu_seqlens_k = cu_seqlens_q.clone()

        max_seqlen_q = int(torch.max(seq_lens).item())
        max_seqlen_k = int(max_seqlen_q)

    key_cache_fa = key_cache_conch.permute(0, 2, 1, 3)
    value_cache_fa = value_cache_conch.permute(0, 2, 1, 3)

    total_num_q = int(cu_seqlens_q[-1].item())

    q = torch.empty(total_num_q, num_query_heads, head_size, dtype=dtype, device=device)
    q.uniform_(-scale, scale)

    vllm_output = flash_attn_varlen_func(
        q=q,
        k=key_cache_fa,
        v=value_cache_fa,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_tables,
        seqused_k=seq_lens,
        softmax_scale=scale,
        causal=causal,
    )

    conch_output = varlen_attention(
        query=q,
        key_cache=key_cache_conch,
        value_cache=value_cache_conch,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        scale=scale,
        softcap=softcap,
    )

    torch.testing.assert_close(vllm_output, conch_output, atol=tolerance, rtol=tolerance)
