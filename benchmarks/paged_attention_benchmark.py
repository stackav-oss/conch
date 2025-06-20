# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch paged attention benchmark."""

from typing import Final

import click
import torch

from conch.ops.attention.paged_attention import paged_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--head-dim",
    required=True,
    type=int,
    default=128,
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
    default=128,
    help="Batch size",
)
@click.option(
    "--num-query-heads",
    required=False,
    type=int,
    default=32,
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
    "--kv-cache-dtype",
    required=False,
    type=click.Choice(["auto", "fp8", "fp8_e4m3"]),
    default="auto",
    help="Dtype of KV cache",
)
@click.option(
    "--iteration-time-ms",
    required=False,
    type=int,
    default=10000,
    help="Time in milliseconds to run benchmark",
)
@click.option(
    "--warmup-time-ms",
    required=False,
    type=int,
    default=1000,
    help="Time in milliseconds to warmup before recording times",
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
    kv_cache_dtype: str,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark PagedAttention.

    Args:
        head_dim: Head dimension of input tensors.
        seq_len: Sequence length of input tensors.
        cache_block_size: Number of KV vectors in each cache block.
        batch_size: Batch size of input tensors.
        num_query_heads: Number of attention query heads.
        num_kv_heads: Number of attention kv heads.
        kv_cache_dtype: Data type of the kv cache.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Conch.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    if kv_cache_dtype != "auto" and not current_platform.supports_fp8():
        error_msg = f"kv_cache_type '{kv_cache_dtype}' not supported on this GPU!"
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

    query, key_cache, value_cache, block_table, seq_lens = create_tensors(
        head_dim,
        seq_len,
        cache_block_size,
        batch_size,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        device,
        dtype,
    )

    scale: Final = float(1.0 / (head_dim**0.5))

    k_scale = torch.full((1,), 0.5, dtype=dtype, device=device)
    v_scale = torch.full((1,), 0.5, dtype=dtype, device=device)

    output_conch = torch.empty_like(query)

    paged_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        output=output_conch,
        scale=scale,
        softcap=0.0,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    conch_result = benchmark_it(
        lambda: paged_attention(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            output=output_conch,
            scale=scale,
            softcap=0.0,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    # Print results
    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
