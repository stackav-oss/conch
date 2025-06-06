# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton rotary_embedding benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.embedding.rotary_embedding import rotary_embedding as rotary_embedding_triton
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
    help="Flag to output results in CSV format",
)
def main(
    head_size: int,
    num_heads: int,
    num_tokens: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton RotaryEmbedding op.

    Args:
        head_size: Head size.
        num_heads: Number of heads.
        num_tokens: Number of tokens.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
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

    query_ref, key_ref = rotary_embedding_reference(
        positions,
        query_ref,
        key_ref,
        cos_sin_cache,
        rotary_dim,
        head_size,
        is_neox_style=is_neox_style,
    )

    query_triton, key_triton = rotary_embedding_triton(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox=is_neox_style,
    )

    if not torch.allclose(query_ref, query_triton, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(query_triton - query_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {query_ref}", file=sys.stderr)
            print(f"Triton output: {query_triton}", file=sys.stderr)
    else:
        print(f"Query matched with atol={absolute_tolerance} :)", file=sys.stderr)

    if not torch.allclose(key_ref, key_triton, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(key_triton - key_ref).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {key_ref}", file=sys.stderr)
            print(f"Triton output: {key_triton}", file=sys.stderr)
    else:
        print(f"Key matched with atol={absolute_tolerance} :)", file=sys.stderr)

    baseline_result = benchmark_it(
        lambda: rotary_embedding_reference(
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
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )
    triton_result = benchmark_it(
        lambda: rotary_embedding_triton(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox=is_neox_style,
        ),
        tag="Triton",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=device,
    )

    # Print results
    triton_result.print_parameters(csv=csv)
    triton_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
