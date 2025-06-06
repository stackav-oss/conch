# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton varlen attention vs. FlashAttnVarlen benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.attention.varlen_attention import varlen_attention
from conch.platforms import current_platform
from conch.third_party.vllm.unified_attention import unified_attention
from conch.third_party.vllm.utils import create_tensors, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # type: ignore[attr-defined, unused-ignore]
else:
    flash_attn_varlen_func = None  # type: ignore[assignment, unused-ignore]


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
    default=1024,
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
    default=10,
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
    default=4,
    help="Number of kv heads",
)
@click.option(
    "--causal",
    is_flag=True,
    help="Flag to toggle causal/non-causal attention",
)
@click.option(
    "--pure-decode",
    is_flag=True,
    help="Flag for making all q_seqlens == 1",
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
    causal: bool,
    pure_decode: bool,
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
        causal: Flag to toggle causal/non-causal attention.
        pure_decode: Flag for making all q_seqlens == 1.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    scale: Final = float(1.0 / (head_dim**0.5))

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "head_dim": head_dim,
            "seq_len": seq_len,
            "cache_block_size": cache_block_size,
            "batch_size": batch_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "causal": causal,
            "pure_decode": pure_decode,
        },
    )

    kv_cache_dtype: Final = "auto"
    dtype: Final = torch.float16

    _, _, _, key_cache, value_cache, block_table, seq_lens = create_tensors(
        head_dim,
        seq_len,
        cache_block_size,
        batch_size,
        num_query_heads,
        num_kv_heads,
        kv_cache_dtype,
        current_platform.device,
        dtype,
    )

    starting_item = torch.as_tensor([0], dtype=torch.int32)

    if pure_decode:
        seqlens_q = torch.ones((batch_size,), dtype=torch.int32)

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

    total_num_q = int(cu_seqlens_q[-1].item())

    query = torch.empty((total_num_q, num_query_heads, head_dim), dtype=dtype, device=device)
    query.uniform_(-scale, scale)

    output_conch = varlen_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
        causal=causal,
    )

    if flash_attn_varlen_func is not None:
        output_vllm = flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            seqused_k=seq_lens,
            softmax_scale=scale,
            causal=causal,
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
            lambda: flash_attn_varlen_func(
                q=query,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                block_table=block_table,
                seqused_k=seq_lens,
                softmax_scale=scale,
                causal=causal,
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

    if causal and flash_attn_varlen_func is None:
        out = torch.zeros_like(query)

        vllm_triton_result = benchmark_it(
            lambda: unified_attention(
                q=query,
                k=key_cache,
                v=value_cache,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seq_lens,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=scale,
                causal=causal,
                window_size=(-1, -1),
                block_table=block_table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
            ),
            tag="vLLM Triton",
            metadata=metadata,
            num_iterations=num_iterations,
            num_warmup_iterations=num_warmup_iterations,
            device=query.device,
        )
    else:
        vllm_triton_result = None

    triton_result = benchmark_it(
        lambda: varlen_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            output=output_conch,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            scale=scale,
            causal=causal,
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
    if vllm_triton_result is not None:
        vllm_triton_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
