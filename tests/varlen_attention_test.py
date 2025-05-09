# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Test Triton varlen attention."""

import math
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
# # MHA, MQA, and GQA
# # - MHA: num_query_heads == num_kv_heads
# # - MQA: num_kv_heads == 1
# # - GQA: num_query_heads != num_kv_heads && num_kv_heads != 1
_NUM_HEADS_ABRIDGED: Final = [(8, 8), (4, 1), (16, 4)]
_SEQUENCE_LENGTHS: Final = [240, 343, 1024]


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


def _attention_forward_core_ref_impl(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, causal: bool, use_exp2: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = q.dtype

    # cast to float32
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Compute attention scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))

    # Scale scores
    attention_scaled_scores = sm_scale * attention_scores

    # Apply causal mask if necessary
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q - L_k
        causal_mask = row_idx >= (col_offset + col_idx)
        # set -inf to places the causal mask is false
        attention_scaled_scores = attention_scaled_scores.masked_fill(
            torch.logical_not(causal_mask.unsqueeze(0)), float("-inf")
        )

    # Compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if causal:
        # Replace -inf in max_scores with zeros to avoid NaN in subtraction
        max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)

    # Shift scores
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores

    # Exponentiate
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    # Sum of exponentials
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if causal:
        # if sum of exp scores is 0.0 it means scores where -inf, we cannot compute softmax and softmax_lse. Setting to 1 deals with -inf case cleanly
        sum_exp_scores = torch.where(sum_exp_scores == 0, torch.ones_like(sum_exp_scores), sum_exp_scores)

    # Compute softmax probabilities
    p = exp_scores / sum_exp_scores

    # Compute log-sum-exp
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)

    # Compute output
    o = torch.matmul(p, v)

    # cast back to original dtype
    o = o.to(original_dtype)

    return o, softmax_lse


def _attention_varlen_forward_pytorch_ref_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    causal: bool,
    layout: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    use_exp2: bool = False,
) -> torch.Tensor:
    # Ensure the layout is 'thd'
    if layout != "thd":
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]

    # Pre-allocate outputs
    total_L_q = q.shape[0]

    o = torch.zeros((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=q.device)

    # Compute group_size for MQA/GQA handling
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = int(cu_seqlens_q[i].item())
        end_q = int(cu_seqlens_q[i + 1].item())
        start_k = int(cu_seqlens_k[i].item())
        end_k = int(cu_seqlens_k[i + 1].item())

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k

        # Extract q_i, k_i, v_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, nheads_q, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, nheads_k, head_dim]

        # Permute to [nheads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)

        # Handle MQA/GQA by adjusting shapes based on group_size
        if group_size != 1:
            # Reshape q_i to [nheads_k, group_size, L_q_i, head_dim]
            q_i = q_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Expand k_i and v_i to match group_size
            k_i = k_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            v_i = v_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            # Flatten the first two dimensions for computation
            q_i = q_i.reshape(nheads_k * group_size, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
        else:
            # Standard case
            q_i = q_i.reshape(nheads_q, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k, seqlen_k, head_dim)

        # Call the core attention function for this sequence
        o_i, _ = _attention_forward_core_ref_impl(q_i, k_i, v_i, sm_scale, causal, use_exp2)

        # Reshape outputs back to original dimensions
        if group_size != 1:
            # Reshape outputs to [nheads_k, group_size, seqlen_q, head_dim]
            o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Combine the first two dimensions back to nheads_q
            o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
        else:
            # Outputs are already in the correct shape
            pass

        # Convert back to 'thd' layout
        o_i = o_i.permute(1, 0, 2)  # [L_q_i, nheads_q, head_dim]

        # Place outputs in pre-allocated tensors
        o[start_q:end_q, :, :] = o_i

    return o


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

    key_cache_fa = key_cache_conch.permute(0, 2, 1, 3)
    value_cache_fa = value_cache_conch.permute(0, 2, 1, 3)

    q = torch.empty(total_num_q, num_query_heads, head_size, dtype=dtype, device=device)
    q.uniform_(-scale, scale)

    pytorch_output = _attention_varlen_forward_pytorch_ref_impl(
        q=q,
        k=key_contiguous,
        v=value_contiguous,
        sm_scale=scale,
        causal=causal,
        layout="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    conch_output = varlen_attention(
        query=q,
        # key_cache=key_cache_conch,
        # value_cache=value_cache_conch,
        key_cache=key_cache_fa,
        value_cache=value_cache_fa,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        softcap=softcap,
        causal=causal,
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
        # key_cache=key_cache_conch,
        # value_cache=value_cache_conch,
        key_cache=key_cache_fa,
        value_cache=value_cache_fa,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        softcap=softcap,
        causal=causal,
    )

    torch.testing.assert_close(vllm_output, conch_output, atol=tolerance, rtol=tolerance)
