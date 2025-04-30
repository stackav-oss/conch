# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Test Triton varlen attention."""

import random
from typing import Final

import pytest
import torch

from conch import envs
from conch.ops.attention.varlen_attention import varlen_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import seed_everything

_ENABLE_VLLM: Final = envs.CONCH_ENABLE_VLLM and current_platform.has_cuda()
# _BATCH_SIZES: Final = [1, 4, 8]
_HEAD_SIZES: Final = [64, 96, 128, 256]
_NUM_SEQS_ABRIDGED: Final = [4, 10]
# _SEQUENCE_LENGTHS: Final = [16, 128, 240, 333, 1002, 2048]
# # MHA, MQA, and GQA
# # - MHA: num_query_heads == num_kv_heads
# # - MQA: num_kv_heads == 1
# # - GQA: num_query_heads != num_kv_heads && num_kv_heads != 1
# _NUM_HEADS: Final = [(8, 8), (4, 4), (8, 1), (4, 1), (16, 4), (16, 2), (8, 4), (8, 2)]
# # Too many parameterizations makes the SDPA test cases too slow
# _SEQUENCE_LENGTHS_ABRIDGED: Final = [240, 333, 1002]
_NUM_HEADS_ABRIDGED: Final = [(8, 8), (4, 1), (16, 4)]
_MAX_SEQLEN_Q: Final = 1024


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


def _create_seqlens(num_seqs: int, different_seqlen_k: bool = False) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Create list of seqlens for query/key."""
    seqlens_q = [0]
    seqlens_k = [0]

    max_seqlen_q = 0
    max_seqlen_k = 0

    for i in range(num_seqs):
        seqlen_q = random.randint(1, _MAX_SEQLEN_Q)
        seqlen_k = seqlen_q + random.randint(1, _MAX_SEQLEN_Q) if different_seqlen_k else seqlen_q

        max_seqlen_q = max(max_seqlen_q, seqlen_q)
        max_seqlen_k = max(max_seqlen_k, seqlen_k)

        seqlens_q.append(seqlen_q)
        seqlens_k.append(seqlen_k)

    return (
        torch.tensor(seqlens_q, dtype=torch.int32),
        torch.tensor(seqlens_k, dtype=torch.int32),
        max_seqlen_q,
        max_seqlen_k,
    )


def _attention_forward_core_ref_impl(
    q, k, v, sm_scale, causal, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2
):
    # cast to float32
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Compute attention scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))

    # Scale scores
    attention_scaled_scores = sm_scale * attention_scores

    # Apply ALiBi if slopes are provided
    if alibi_slopes is not None:
        L_q, L_k = q.shape[1], k.shape[1]
        alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
        alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
        attention_scaled_scores = attention_scaled_scores + alibi_bias

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

    # apply dropout if specified
    if dropout_p > 0.0:
        rand_vals = torch.rand(
            p.shape, generator=torch.Generator(device=p.device).manual_seed(philox_seed), device=p.device, dtype=p.dtype
        )
        dropout_mask, dropout_scale = rand_vals > dropout_p, (1.0 / (1 - dropout_p))
        # Apply dropout mask and scale
        # Set -1 for dropped positions and 1 for kept positions in exp_scores
        sd_mask = torch.where(dropout_mask, exp_scores, -exp_scores)
        p = torch.where(dropout_mask, p, torch.zeros_like(p)) * dropout_scale
    else:
        sd_mask = exp_scores

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
    o = o.to(torch.float16)
    # softmax_lse = softmax_lse.to(torch.float16) # NOTE: if you cast lse to fp16 it cause accuracy issues. keep fp32
    sd_mask = sd_mask.to(torch.float16)

    return o, softmax_lse, sd_mask


def _attention_varlen_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    philox_seed=None,
    philox_offset=None,
    alibi_slopes=None,
    use_exp2=False,
):
    # Ensure the layout is 'thd'
    if layout != "thd":
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]

    # Pre-allocate outputs
    total_L_q = q.shape[0]
    total_L_k = k.shape[0]

    o = torch.zeros((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=q.device)
    # softmax_lse = torch.zeros((total_L_q, nheads_q), dtype=torch.float32, device=q.device)
    # sd_mask = torch.zeros((batch_size, nheads_q, max_seqlen_q, max_seqlen_k), dtype=torch.float32, device=q.device)

    # Compute group_size for MQA/GQA handling
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError("nheads_q must be divisible by nheads_k")

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

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

        if alibi_slopes is not None:
            alibi_slopes_i = alibi_slopes[i]
        else:
            alibi_slopes_i = None

        # Call the core attention function for this sequence
        o_i, softmax_lse_i, sd_mask_i = _attention_forward_core_ref_impl(
            q_i, k_i, v_i, sm_scale, causal, dropout_p, philox_seed, philox_offset, alibi_slopes_i, use_exp2
        )

        # Reshape outputs back to original dimensions
        if group_size != 1:
            # Reshape outputs to [nheads_k, group_size, seqlen_q, head_dim]
            o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            # Combine the first two dimensions back to nheads_q
            o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
            # Reshape softmax_lse_i similarly
            # softmax_lse_i = softmax_lse_i.reshape(nheads_k, group_size, seqlen_q)
            # softmax_lse_i = softmax_lse_i.reshape(nheads_q, seqlen_q)
        else:
            # Outputs are already in the correct shape
            pass

        # Convert back to 'thd' layout
        o_i = o_i.permute(1, 0, 2)  # [L_q_i, nheads_q, head_dim]
        # softmax_lse_i = softmax_lse_i.permute(1, 0)  # [L_q_i, nheads_q]
        # sd_mask_i = sd_mask_i # [nheads_q, L_q_i, L_k_i]

        # Place outputs in pre-allocated tensors
        o[start_q:end_q, :, :] = o_i
        # softmax_lse[start_q:end_q, :] = softmax_lse_i
        # sd_mask[i, :, :seqlen_q, :seqlen_k] = sd_mask_i

    # return o, softmax_lse, sd_mask
    return o


@pytest.mark.skipif(not _ENABLE_VLLM, reason="This test case requires vLLM")
# @pytest.mark.parametrize("num_seqs", _NUM_SEQS_ABRIDGED)
@pytest.mark.parametrize("num_seqs", [1])
# @pytest.mark.parametrize("head_size", _HEAD_SIZES)
@pytest.mark.parametrize("head_size", [128])
# @pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS_ABRIDGED)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), [(8, 8)])
@pytest.mark.parametrize("different_seqlen_k", [False])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_varlen_attention_vs_flash(
    num_seqs: int,
    head_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    different_seqlen_k: bool,
    dtype: torch.dtype,
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

    seqlen_q, seqlen_k, max_seqlen_q, max_seqlen_k = _create_seqlens(num_seqs, different_seqlen_k)

    cu_seqlen_q = torch.cumsum(seqlen_q, dim=0, dtype=torch.int32)
    cu_seqlen_k = torch.cumsum(seqlen_k, dim=0, dtype=torch.int32)
    total_num_q = torch.sum(cu_seqlen_q, dim=0).item()
    total_num_k = torch.sum(cu_seqlen_k, dim=0).item()

    print(f"{num_seqs = }")
    print(f"{different_seqlen_k = }")
    print(f"{cu_seqlen_q = }")
    print(f"{cu_seqlen_k = }")
    print(f"{max_seqlen_q = }")
    print(f"{max_seqlen_k = }")
    print(f"{total_num_q = }")
    print(f"{total_num_k = }")

    q = torch.randn(total_num_q, num_query_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(total_num_k, num_kv_heads, head_size, device=device, dtype=dtype)
    v = torch.randn(total_num_k, num_kv_heads, head_size, device=device, dtype=dtype)

    print(f"{q = }")
    print(f"{k = }")
    print(f"{v = }")

    pytorch_output = _attention_varlen_forward_pytorch_ref_impl(
        q=q,
        k=k,
        v=v,
        sm_scale=scale,
        causal=True,
        layout="thd",
        cu_seqlens_q=cu_seqlen_q,
        cu_seqlens_k=cu_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    vllm_output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlen_q,
        cu_seqlens_k=cu_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        softcap=softcap,
    )

    conch_output = varlen_attention(
        query=q,
        key=k,
        value=v,
        cu_seqlen_q=cu_seqlen_q,
        cu_seqlen_k=cu_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        softcap=softcap,
    )

    print(f"{pytorch_output = }")
    print(f"{vllm_output = }")
    print(f"{conch_output = }")

    torch.testing.assert_close(vllm_output, pytorch_output, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(vllm_output, conch_output, atol=tolerance, rtol=tolerance)
