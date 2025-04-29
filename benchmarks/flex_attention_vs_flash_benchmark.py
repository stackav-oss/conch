# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""FlashAttn vs. Flex Attention benchmark."""

import sys
from typing import Final

import click
import torch
from torch.nn.attention.flex_attention import flex_attention

from conch import envs
from conch.platforms import current_platform
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it

if envs.CONCH_ENABLE_VLLM and current_platform.is_nvidia():
    from vllm.vllm_flash_attn import flash_attn_func  # type: ignore[attr-defined, unused-ignore]
else:
    flash_attn_func = None  # type: ignore[assignment]


@click.command()
@click.option(
    "-h",
    "--head-dim",
    required=True,
    type=int,
    # default=256,
    default=128,
    help="Head dimension",
)
@click.option(
    "-s",
    "--seq-len",
    required=True,
    type=int,
    # default=8192,
    # default=2048,
    default=1024,
    help="Sequence length (for k/v)",
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
    """Benchmark FlexAttention.

    Args:
        head_dim: Head dimension of input tensors.
        seq_len: Sequence length of input tensors.
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
    if not current_platform.is_nvidia() or flash_attn_func is None:
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
            "batch_size": batch_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
        },
    )

    q = torch.randn(
        batch_size, num_query_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=False
    )

    q_vllm = q.detach().clone().permute(0, 2, 1, 3)
    k_vllm = k.detach().clone().permute(0, 2, 1, 3)
    v_vllm = v.detach().clone().permute(0, 2, 1, 3)

    scale: Final = float(1.0 / (head_dim**0.5))

    enable_gqa = (num_query_heads != num_kv_heads)

    # Check accuracy match
    output_vllm = flash_attn_func(
        q_vllm,
        k_vllm,
        v_vllm,
        softmax_scale=scale,
        causal=True,
    )

    output_vllm = output_vllm.permute(0, 2, 1, 3)

    flex_attention_c = torch.compile(flex_attention)

    output_flex = flex_attention_c(q, k, v, scale=scale, enable_gqa=enable_gqa)

    print(f"{output_vllm.shape = }")
    print(f"{output_flex.shape = }")

    if not torch.allclose(output_vllm, output_flex, atol=absolute_tolerance):
        print(f"WARNING: Reference and Flex results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(output_flex - output_vllm).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {output_vllm}", file=sys.stderr)
            print(f"Flex output: {output_flex}", file=sys.stderr)
    else:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: flash_attn_func(
            q_vllm,
            k_vllm,
            v_vllm,
            softmax_scale=scale,
            causal=True,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    flex_result = benchmark_it(
        lambda: flex_attention_c(
            q,
            k,
            v,
            scale=scale,
            enable_gqa=enable_gqa,
        ),
        tag="Flex",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    # Print results
    flex_result.print_parameters(csv=csv)
    flex_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
