# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton paged attention benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.attention.paged_attention import paged_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
    from vllm._custom_ops import paged_attention_v2 as vllm_paged_attention_v2
else:
    vllm_paged_attention_v2 = None  # type: ignore[assignment, unused-ignore]


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
    "--kv-cache-dtype",
    required=False,
    type=click.Choice(["auto", "fp8", "fp8_e4m3"]),
    default="auto",
    help="Dtype of KV cache",
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
    kv_cache_dtype: str,
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
        kv_cache_dtype: Data type of the kv cache.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
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

    query, key_cache_vllm, value_cache_vllm, key_cache_conch, value_cache_conch, block_table, seq_lens = create_tensors(
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
    max_seq_len: Final = int(seq_lens.max().item())

    k_scale = torch.full((1,), 0.5, dtype=dtype, device=device)
    v_scale = torch.full((1,), 0.5, dtype=dtype, device=device)

    output_conch = torch.empty_like(query)

    paged_attention(
        query,
        key_cache_conch,
        value_cache_conch,
        block_table,
        seq_lens,
        output=output_conch,
        scale=scale,
        softcap=0.0,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    if vllm_paged_attention_v2 is not None:
        # VLLM scratchpads
        partition_size: Final = 512
        max_num_partitions = (max_seq_len + partition_size - 1) // partition_size
        tmp_output = torch.empty(
            size=(batch_size, num_query_heads, max_num_partitions, head_dim),
            dtype=query.dtype,
            device=query.device,
        )
        exp_sums = torch.empty(
            size=(batch_size, num_query_heads, max_num_partitions),
            dtype=torch.float32,
            device=query.device,
        )
        max_logits = torch.empty_like(exp_sums)

        alibi_slopes = None

        output_vllm = torch.empty_like(query)

        # Check accuracy match
        vllm_paged_attention_v2(
            output_vllm,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache_vllm,
            value_cache_vllm,
            num_kv_heads,
            scale,
            block_table,
            seq_lens,
            cache_block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
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
            lambda: vllm_paged_attention_v2(
                output_vllm,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache_vllm,
                value_cache_vllm,
                num_kv_heads,
                scale,
                block_table,
                seq_lens,
                cache_block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            ),
            tag="Baseline",
            metadata=metadata,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            device=query.device,
        )
    else:
        print("Skipping checking vs. reference vLLM implementation...", file=sys.stderr)
        baseline_result = None

    triton_result = benchmark_it(
        lambda: paged_attention(
            query,
            key_cache_conch,
            value_cache_conch,
            block_table,
            seq_lens,
            output=output_conch,
            scale=scale,
            softcap=0.0,
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
    if baseline_result is not None:
        baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
