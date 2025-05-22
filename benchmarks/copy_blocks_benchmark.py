# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton copy_blocks benchmark."""

import random
import sys
from typing import Final

import click
import torch

from conch.ops.vllm.copy_blocks import copy_blocks as copy_blocks_triton
from conch.platforms import current_platform
from conch.reference.vllm.copy_blocks import copy_blocks as copy_blocks_reference
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


@click.command()
@click.option(
    "--head-dim",
    required=True,
    type=int,
    default=256,
    help="Head dimension",
)
@click.option(
    "--num-layers",
    required=True,
    type=int,
    default=8,
    help="Number of layers",
)
@click.option(
    "--cache-block-size",
    required=True,
    type=int,
    default=32,
    help="Number of KV vectors in each cache block",
)
@click.option(
    "--num-kv-heads",
    required=False,
    type=int,
    default=8,
    help="Number of kv heads",
)
@click.option(
    "--num-blocks",
    required=False,
    type=int,
    default=2000,
    help="Number of blocks in cache",
)
@click.option(
    "--num-mappings",
    required=False,
    type=int,
    default=512,
    help="Number of mappings to copy",
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
    num_layers: int,
    cache_block_size: int,
    num_kv_heads: int,
    num_blocks: int,
    num_mappings: int,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton copy_blocks operation.

    Args:
        head_dim: Head dimension of input tensors.
        num_layers: Sequence length of input tensors.
        cache_block_size: Number of KV vectors in each cache block.
        num_kv_heads: Number of attention kv heads.
        num_blocks: Number of blocks in the cache.
        num_mappings: Number of pairs of blocks to copy.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    dtype: Final = torch.float16

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "head_dim": head_dim,
            "num_layers": num_layers,
            "cache_block_size": cache_block_size,
            "num_kv_heads": num_kv_heads,
            "num_blocks": num_blocks,
            "num_mappings": num_mappings,
        },
    )

    # Generate random block mappings where each source block is mapped to two
    # destination blocks.
    assert 2 * num_mappings <= num_blocks
    src_blocks = random.sample(range(num_blocks), num_mappings)
    remainig_blocks = list(set(range(num_blocks)) - set(src_blocks))
    dst_blocks = random.sample(remainig_blocks, 2 * num_mappings)
    block_mapping: list[tuple[int, int]] = []
    for i in range(num_mappings):
        src = src_blocks[i]
        dst1 = dst_blocks[2 * i]
        dst2 = dst_blocks[2 * i + 1]
        block_mapping.append((src, dst1))
        block_mapping.append((src, dst2))

    # Create the KV caches.
    key_caches = [
        torch.randn((num_blocks, cache_block_size * num_kv_heads * head_dim), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]
    value_caches = [
        torch.randn((num_blocks, cache_block_size * num_kv_heads * head_dim), dtype=dtype, device=device)
        for _ in range(num_layers)
    ]

    # Clone the KV caches (must happen before calling either kernel bc modifications happen inplace).
    cloned_key_caches = [key_cache.clone() for key_cache in key_caches]
    cloned_value_caches = [value_cache.clone() for value_cache in value_caches]

    # Convert mapping list to tensor
    block_mapping_tensor = torch.tensor(block_mapping, dtype=torch.int64, device=device).view(-1, 2)

    # Run the reference implementation.
    copy_blocks_reference(cloned_key_caches, cloned_value_caches, block_mapping)
    # Call Triton kernel
    copy_blocks_triton(key_caches, value_caches, block_mapping_tensor)

    # Compare the results.
    num_key_matched = 0
    num_value_matched = 0

    for key_cache, cloned_key_cache in zip(key_caches, cloned_key_caches, strict=False):
        if not torch.allclose(key_cache, cloned_key_cache, atol=absolute_tolerance):
            print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
            print(f"Output max diff: {(cloned_key_cache - key_cache).abs().max().item()}", file=sys.stderr)

            if verbose:
                print(f"Reference output: {key_cache}", file=sys.stderr)
                print(f"Triton output: {cloned_key_cache}", file=sys.stderr)
        else:
            num_key_matched += 1

    for value_cache, cloned_value_cache in zip(value_caches, cloned_value_caches, strict=False):
        if not torch.allclose(value_cache, cloned_value_cache, atol=absolute_tolerance):
            print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
            print(f"Output max diff: {(cloned_value_cache - value_cache).abs().max().item()}", file=sys.stderr)

            if verbose:
                print(f"Reference output: {value_cache}", file=sys.stderr)
                print(f"Triton output: {cloned_value_cache}", file=sys.stderr)
        else:
            num_value_matched += 1

    if num_key_matched == num_layers and num_value_matched == num_layers:
        print(f"Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. Triton implementations
    baseline_result = benchmark_it(
        lambda: copy_blocks_reference(
            cloned_key_caches,
            cloned_value_caches,
            block_mapping,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=block_mapping_tensor.device,
    )

    triton_result = benchmark_it(
        lambda: copy_blocks_triton(
            key_caches,
            value_caches,
            block_mapping_tensor,
        ),
        tag="Triton",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=block_mapping_tensor.device,
    )

    # Print results
    triton_result.print_parameters(csv=csv)
    triton_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
