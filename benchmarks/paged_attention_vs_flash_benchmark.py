# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton paged attention vs. FlashAttnWithKVCache benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.kernels.attention.paged_attention import MAX_NUM_SPLITS, paged_attention_launcher
from conch.ops.attention.paged_attention import split_kv_cache
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
    from vllm.vllm_flash_attn import flash_attn_with_kvcache  # type: ignore[attr-defined, unused-ignore]
else:
    flash_attn_with_kvcache = None  # type: ignore[assignment]


@click.command()
@click.option(
    "-h",
    "--head-dim",
    required=True,
    type=int,
    default=256,
    help="Head dimension",
)
@click.option(
    "-s",
    "--seq-len",
    required=True,
    type=int,
    default=8192,
    help="Sequence length (for k/v)",
)
@click.option(
    "-c",
    "--cache-block-size",
    required=True,
    type=int,
    default=32,
    help="Number of KV vectors in each cache block",
)
@click.option(
    "-b",
    "--batch-size",
    required=False,
    type=int,
    default=4,
    help="Batch size",
)
@click.option(
    "-h",
    "--num-query-heads",
    required=False,
    type=int,
    default=8,
    help="Number of query heads",
)
@click.option(
    "-k",
    "--num-kv-heads",
    required=False,
    type=int,
    default=8,
    help="Number of kv heads",
)
@click.option(
    "-i",
    "--num-iterations",
    required=False,
    type=int,
    default=100,
    help="Number of iterations",
)
@click.option(
    "-w",
    "--num-warmup-iterations",
    required=False,
    type=int,
    default=10,
    help="Number of warmup iterations",
)
@click.option(
    "-a",
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-3,
    help="Absolute tolerance to match with",
)
@click.option(
    "-v",
    "--verbose",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Flag for printing verbose output",
)
@click.option(
    "-g",
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
@click.option(
    "--csv",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
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

    query, _, _, key_cache_conch, value_cache_conch, block_tables, seq_lens = create_tensors(
        head_dim, seq_len, cache_block_size, batch_size, num_query_heads, num_kv_heads, "auto", gpu, torch.float16
    )

    _, max_num_blocks_per_seq = block_tables.shape

    scale: Final = float(1.0 / (head_dim**0.5))

    # Allocate additional memory for intermediate result (of shape (head_dim,)) for each batch/split/query head
    output_scratchpad = torch.zeros(
        (batch_size, MAX_NUM_SPLITS, num_query_heads, head_dim), dtype=query.dtype, device=query.device
    )

    # # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/split/query head
    lse_scratchpad = torch.zeros((batch_size, MAX_NUM_SPLITS, num_query_heads), dtype=query.dtype, device=query.device)

    alibi_slopes = None

    kv_cache_conch = torch.vstack((key_cache_conch[None, :, :], value_cache_conch[None, :, :]))
    key_cache_conch, value_cache_conch = split_kv_cache(kv_cache_conch, num_kv_heads, head_dim)

    # Create output tensors
    query_vllm = query.unsqueeze(1)
    output_conch = torch.empty_like(query)

    key_cache_vllm = key_cache_conch.permute(0, 2, 1, 3)
    value_cache_vllm = value_cache_conch.permute(0, 2, 1, 3)

    softcap = 30.0

    # Check accuracy match
    output_vllm = flash_attn_with_kvcache(
        query_vllm,
        key_cache_vllm,
        value_cache_vllm,
        block_table=block_tables,
        cache_seqlens=seq_lens,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        softcap=softcap,
    )

    output_vllm = output_vllm.squeeze(1)

    paged_attention_launcher(
        output_conch,
        query,
        key_cache_conch,
        value_cache_conch,
        output_scratchpad,
        lse_scratchpad,
        scale,
        block_tables,
        seq_lens,
        softcap=softcap,
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
            key_cache_vllm,
            value_cache_vllm,
            block_table=block_tables,
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
        lambda: paged_attention_launcher(
            output_conch,
            query,
            key_cache_conch,
            value_cache_conch,
            output_scratchpad,
            lse_scratchpad,
            scale,
            block_tables,
            seq_lens,
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
