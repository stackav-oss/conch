# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton paged attention benchmark."""

from typing import Final

import click
import torch

from conch import envs
from conch.kernels.attention.paged_attention import MAX_NUM_SPLITS, paged_attention_launcher
from conch.ops.attention.paged_attention import split_kv_cache
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors
from conch.utils.benchmark import benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
    from vllm._custom_ops import paged_attention_v2 as vllm_paged_attention_v2
else:
    vllm_paged_attention_v2 = None  # type: ignore[assignment]


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
    default=2048,
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
    "-d",
    "--kv-cache-dtype",
    required=False,
    type=click.Choice(["auto", "fp8", "fp8_e4m3"]),
    default="auto",
    help="Dtype of KV cache",
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
    """
    if kv_cache_dtype != "auto" and not current_platform.supports_fp8():
        error_msg = f"kv_cache_type '{kv_cache_dtype}' not supported on this GPU!"
        raise NotImplementedError(error_msg)

    print(f"{head_dim=}, {seq_len=}, {cache_block_size=}, {batch_size=}, {num_query_heads=}, {num_kv_heads=}")

    query, key_cache_vllm, value_cache_vllm, key_cache_conch, value_cache_conch, block_tables, seq_lens = (
        create_tensors(
            head_dim,
            seq_len,
            cache_block_size,
            batch_size,
            num_query_heads,
            num_kv_heads,
            kv_cache_dtype,
            gpu,
            torch.float16,
        )
    )

    _, max_num_blocks_per_seq = block_tables.shape

    scale: Final = float(1.0 / (head_dim**0.5))
    max_seq_len: Final = int(seq_lens.max().item())

    # Allocate additional memory for intermediate result (of shape (head_dim,)) for each batch/split/query head
    output_scratchpad = torch.zeros(
        (batch_size, MAX_NUM_SPLITS, num_query_heads, head_dim), dtype=query.dtype, device=query.device
    )

    # # Allocate additional memory for intermediate log-sum-exp ("lse", scalar value per-cache block) for each batch/split/query head
    lse_scratchpad = torch.zeros((batch_size, MAX_NUM_SPLITS, num_query_heads), dtype=query.dtype, device=query.device)

    kv_cache_conch = torch.vstack((key_cache_conch[None, :, :], value_cache_conch[None, :, :]))
    key_cache_conch, value_cache_conch = split_kv_cache(kv_cache_conch, num_kv_heads, head_dim)

    # Using default kv_scale
    k_scale = v_scale = 1.0

    output_conch = torch.empty_like(query)

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
            block_tables,
            seq_lens,
            cache_block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

        if not torch.allclose(output_vllm, output_conch, atol=absolute_tolerance):
            print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})")
            print(f"Output max diff: {(output_conch - output_vllm).abs().max().item()}")

            if verbose:
                print(f"Reference output: {output_vllm}")
                print(f"Triton output: {output_conch}")
        else:
            print(f"Results matched with atol={absolute_tolerance} :)")

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
                block_tables,
                seq_lens,
                cache_block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
            ),
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            device=query.device,
        )

        baseline_result.pretty_print(name="Baseline", unit="ms")
    else:
        print("Skipping checking vs. reference vLLM implementation...")

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
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=query.device,
    )

    triton_result.pretty_print(name="Triton", unit="ms")


if __name__ == "__main__":
    main()
