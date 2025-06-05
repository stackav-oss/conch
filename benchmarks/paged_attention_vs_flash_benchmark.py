# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton paged attention vs. FlashAttnWithKVCache benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.attention.paged_attention import paged_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
    from vllm.vllm_flash_attn import flash_attn_with_kvcache  # type: ignore[attr-defined, unused-ignore]
else:
    flash_attn_with_kvcache = None  # type: ignore[assignment, unused-ignore]


@click.command()
@click.option(
    "--head-dim",
    required=True,
    type=int,
    default=256,
    help="Head dimension",
)
@click.option(
    "--seq-len",
    required=True,
    type=int,
    default=8192,
    help="Sequence length (for k/v)",
)
@click.option(
    "--cache-block-size",
    required=True,
    type=int,
    default=32,
    help="Number of KV vectors in each cache block",
)
@click.option(
    "--batch-size",
    required=False,
    type=int,
    default=4,
    help="Batch size",
)
@click.option(
    "--num-query-heads",
    required=False,
    type=int,
    default=8,
    help="Number of query heads",
)
@click.option(
    "--num-kv-heads",
    required=False,
    type=int,
    default=8,
    help="Number of kv heads",
)
@click.option(
    "--num-iterations",
    required=False,
    type=int,
    default=100,
    help="Number of iterations",
)
@click.option(
    "--num-warmup-iterations",
    required=False,
    type=int,
    default=10,
    help="Number of warmup iterations",
)
@click.option(
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-3,
    help="Absolute tolerance to match with",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Flag for printing verbose output",
)
@click.option(
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
@click.option(
    "--csv",
    is_flag=True,
    help="Flag for printing results in CSV format",
)
def main(
    head_dim: int,
    seq_len: int,
    cache_block_size: int,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton PagedAttention.

    Args:
        head_dim: Head dimension of input tensors.
        seq_len: Sequence length of input tensors.
        cache_block_size: Number of KV vectors in each cache block.
        batch_size: Batch size of input tensors.
        num_query_heads: Number of attention query heads.
        num_kv_heads: Number of attention kv heads.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    if not current_platform.is_nvidia() or flash_attn_with_kvcache is None:
        error_msg = "Platform must be Nvidia and vLLM must be installed & enabled via CONCH_ENABLE_VLLM=1"
        raise NotImplementedError(error_msg)

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "head_dim": head_dim,
            "seq_len": seq_len,
            "cache_block_size": cache_block_size,
            "batch_size": batch_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
        },
    )

    kv_cache_dtype = "auto"

    query, _, _, key_cache, value_cache, block_table, seq_lens = create_tensors(
        head_dim, seq_len, cache_block_size, batch_size, num_query_heads, num_kv_heads, kv_cache_dtype, device, dtype
    )

    _, max_num_blocks_per_seq = block_table.shape

    scale: Final = float(1.0 / (head_dim**0.5))

    alibi_slopes = None

    # Create output tensors
    query_vllm = query.unsqueeze(1)
    output_conch = torch.empty_like(query)

    softcap = 30.0
    k_scale = torch.full((1,), 1.0, dtype=dtype, device=device)
    v_scale = torch.full((1,), 1.0, dtype=dtype, device=device)

    # Check accuracy match
    output_vllm = flash_attn_with_kvcache(
        query_vllm,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=seq_lens,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )

    output_vllm = output_vllm.squeeze(1)

    paged_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        output=output_conch,
        scale=scale,
        softcap=softcap,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    if not torch.allclose(output_vllm, output_conch, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(output_conch - output_vllm).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {output_vllm}", file=sys.stderr)
            print(f"Triton output: {output_conch}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: flash_attn_with_kvcache(
            query_vllm,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=seq_lens,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=alibi_slopes,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=query.device,
    )

    triton_result = benchmark_it(
        lambda: paged_attention(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            output=output_conch,
            scale=scale,
            softcap=softcap,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        tag="Triton",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=query.device,
    )

    # Print results
    triton_result.print_parameters(csv=csv)
    triton_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
