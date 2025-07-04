# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch rotary_embedding benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.embedding.rotary_embedding import rotary_embedding as rotary_embedding_conch
from conch.platforms import current_platform
from conch.reference.embedding.rotary_embedding import compute_cos_sin_cache
from conch.reference.embedding.rotary_embedding import rotary_embedding as rotary_embedding_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--head-size",
    required=False,
    type=int,
    default=256,
    help="Feedforward hidden size",
)
@click.option(
    "--num-heads",
    required=False,
    type=int,
    default=8,
)
@click.option(
    "--num-tokens",
    required=False,
    type=int,
    default=8192,
    help="Number of tokens",
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
    help="Flag to output results in CSV format",
)
@click.option(
    "--compile-ref",
    is_flag=True,
    help="Flag to torch.compile() the reference impl",
)
@click.option(
    "--compile-conch",
    is_flag=True,
    help="Flag to torch.compile() the Conch impl",
)
def main(
    head_size: int,
    num_heads: int,
    num_tokens: int,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    compile_conch: bool,
) -> None:
    """Benchmark Conch RotaryEmbedding op.

    Args:
        head_size: Head size.
        num_heads: Number of heads.
        num_tokens: Number of tokens.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Conch.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
        compile_ref: Flag to torch.compile() the reference implementation.
        compile_conch: Flag to torch.compile() the Conch implementation.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    base: Final = 10000
    is_neox_style: Final = True
    rotary_dim: Final = head_size
    max_position = num_tokens

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "head_size": head_size,
            "num_heads": num_heads,
            "num_tokens": num_tokens,
        },
    )

    cos_sin_cache = compute_cos_sin_cache(base, rotary_dim, max_position)

    positions = torch.randint(0, max_position, (num_tokens,))
    query = torch.randn(num_tokens, num_heads * head_size)
    key = torch.randn_like(query)
    # Need to clone since updates are performed in-place
    query_ref = torch.clone(query)
    key_ref = torch.clone(key)

    rotary_embedding_conch_fn = torch.compile(rotary_embedding_conch) if compile_conch else rotary_embedding_conch
    rotary_embedding_ref_fn = torch.compile(rotary_embedding_reference) if compile_ref else rotary_embedding_reference

    query_ref, key_ref = rotary_embedding_ref_fn(
        positions,
        query_ref,
        key_ref,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style=is_neox_style,
    )

    query_conch, key_conch = rotary_embedding_conch_fn(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
    )

    if not torch.allclose(query_ref, query_conch, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(query_conch - query_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {query_ref}", file=sys.stderr)
            print(f"Conch output: {query_conch}", file=sys.stderr)
    else:
        print(f"Query matched with atol={absolute_tolerance} :)", file=sys.stderr)

    if not torch.allclose(key_ref, key_conch, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(key_conch - key_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {key_ref}", file=sys.stderr)
            print(f"Conch output: {key_conch}", file=sys.stderr)
    else:
        print(f"Key matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: rotary_embedding_ref_fn(
            positions,
            query_ref,
            key_ref,
            cos_sin_cache,
            rotary_dim,
            head_size,
            is_neox_style=is_neox_style,
        ),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )
    conch_result = benchmark_it(
        lambda: rotary_embedding_conch_fn(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox=is_neox_style,
        ),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    # Print results
    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
