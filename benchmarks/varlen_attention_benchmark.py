# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton varlen attention vs. FlashAttnVarlen benchmark."""

import sys
from typing import Final

import click
import torch

from conch import envs
from conch.ops.attention.varlen_attention import varlen_attention
from conch.platforms import current_platform
from conch.third_party.vllm.utils import create_tensors, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # type: ignore[attr-defined, unused-ignore]
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


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
    default=1024,
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
    default=10,
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
    default=4,
    help="Number of kv heads",
)
@click.option(
    "--causal",
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Flag for printing results in CSV format",
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
    causal: bool,
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
    if not current_platform.is_nvidia() or flash_attn_varlen_func is None:
        error_msg = "Platform must be Nvidia and vLLM must be installed & enabled via CONCH_ENABLE_VLLM=1"
        raise NotImplementedError(error_msg)

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
        },
    )

    tolerance: Final = 1e-3
    kv_cache_dtype: Final = "auto"
    dtype: Final = torch.float16

    _, _, _, key_cache_conch, value_cache_conch, block_tables, seq_lens = create_tensors(
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

    cu_seqlens_q = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    # cu_seqlens_k = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

    cu_seqlens_q = torch.cat((starting_item, cu_seqlens_q), dim=0)
    cu_seqlens_k = cu_seqlens_q.clone()
    # cu_seqlens_k = torch.cat((starting_item, cu_seqlens_k), dim=0)

    max_seqlen_q = torch.max(seq_lens).item()
    max_seqlen_k = max_seqlen_q

    # print(f"{seq_lens = }")
    # print(f"{cu_seqlens_q = }")
    # print(f"{cu_seqlens_k = }")

    key_cache_fa = key_cache_conch.permute(0, 2, 1, 3)
    value_cache_fa = value_cache_conch.permute(0, 2, 1, 3)

    # q = torch.empty(cu_seqlens_q[-1], num_query_heads, head_dim, dtype=dtype, device=device)
    query = torch.empty(cu_seqlens_q[-1], num_query_heads, head_dim, dtype=dtype, device=device)
    query.uniform_(-scale, scale)

    # query, _, _, key_cache_conch, value_cache_conch, block_tables, seq_lens = create_tensors(
    #     head_dim, seq_len, cache_block_size, batch_size, num_query_heads, num_kv_heads, "auto", gpu, torch.float16
    # )

    # _, max_num_blocks_per_seq = block_tables.shape

    # scale: Final = float(1.0 / (head_dim**0.5))

    alibi_slopes = None

    # Create output tensors
    # query_vllm = query.unsqueeze(1)
    output_conch = torch.empty_like(query)

    key_cache_vllm = key_cache_conch.permute(0, 2, 1, 3)
    value_cache_vllm = value_cache_conch.permute(0, 2, 1, 3)

    # k_scale = torch.full((1,), 1.0)
    # v_scale = torch.full((1,), 1.0)

    output_vllm = flash_attn_varlen_func(
        q=query,
        k=key_cache_fa,
        v=value_cache_fa,
        cu_seqlens_q=cu_seqlens_q,
        # cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_table=block_tables,
        seqused_k=seq_lens,
        softmax_scale=scale,
        causal=causal,
    )

    output_conch = varlen_attention(
        query=query,
        key_cache=key_cache_conch,
        value_cache=value_cache_conch,
        block_tables=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=scale,
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
            k=key_cache_fa,
            v=value_cache_fa,
            cu_seqlens_q=cu_seqlens_q,
            # cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_table=block_tables,
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

    triton_result = benchmark_it(
        lambda: varlen_attention(
            query=query,
            key_cache=key_cache_conch,
            value_cache=value_cache_conch,
            block_tables=block_tables,
            seq_lens=seq_lens,
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
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
