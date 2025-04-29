# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Test Triton varlen attention."""

from typing import Final

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from einops import einsum, rearrange

from conch import envs
from conch.ops.attention.varlen_attention import varlen_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import seed_everything

_ENABLE_VLLM: Final = envs.CONCH_ENABLE_VLLM and current_platform.has_cuda()
_BATCH_SIZES: Final = [1, 4, 8]
_HEAD_SIZES: Final = [64, 96, 128, 256]
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

    q = torch.randn(
        batch_size, num_query_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, num_kv_heads, sequence_length, head_size, device=device, dtype=dtype, requires_grad=False
    )

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

    # final_out_sdpa = out_sdpa[:, :, -1, :]

    q_paged = q[:, :, -1, :]
    out_paged = torch.zeros_like(q_paged, dtype=dtype, device=device)

    paged_attention(
        out_paged,
        q_paged,
        torch.vstack((key_cache_paged[None, :, :], value_cache_paged[None, :, :])),
        num_kv_heads,
        scale,
        block_tables,
        sequence_lengths,
        cache_block_size,
    )

    torch.testing.assert_close(final_out_sdpa, out_paged, atol=atol, rtol=rtol)


@pytest.mark.parametrize("batch_size", _BATCH_SIZES_ABRIDGED)
@pytest.mark.parametrize("head_size", _HEAD_SIZES_VLLM)
@pytest.mark.parametrize(("num_query_heads", "num_kv_heads"), _NUM_HEADS_ABRIDGED)
@pytest.mark.parametrize("sequence_length", _SEQUENCE_LENGTHS_ABRIDGED)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_paged_attention(
    batch_size: int,
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
        dtype,
        gqa_workaround,
    )
