# Copyright (C) 2025 Stack AV Co. - All Rights Reserved.

"""Triton reshape_and_cache benchmark."""

import random
import sys
from typing import Final

import click
import torch

from conch.ops.attention.paged_attention import split_kv_cache
from conch.ops.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_triton
from conch.platforms import current_platform
from conch.reference.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_reference
from conch.third_party.vllm.utils import create_kv_caches_with_random, reshape_vllm_kvcache, seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _to_conch_layout(
    key_cache: torch.Tensor, value_cache: torch.Tensor, num_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert caches in vLLM layout to Conch layout."""
    k, v = reshape_vllm_kvcache(key_cache, value_cache)
    kv_cache = torch.vstack((k[None, :, :], v[None, :, :]))
    return split_kv_cache(kv_cache, num_heads, head_size)


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
    "-t",
    "--num-tokens",
    required=True,
    type=int,
    default=512,
    help="Number of tokens",
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
    "-k",
    "--num-kv-heads",
    required=False,
    type=int,
    default=8,
    help="Number of kv heads",
)
@click.option(
    "-b",
    "--num-blocks",
    required=False,
    type=int,
    default=2000,
    help="Number of blocks in cache",
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
    num_tokens: int,
    cache_block_size: int,
    num_kv_heads: int,
    num_blocks: int,
    kv_cache_dtype: str,
    num_iterations: int,
    num_warmup_iterations: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
) -> None:
    """Benchmark Triton reshape_and_cache.

    Args:
        head_dim: Head dimension of input tensors.
        num_tokens: Number of tokens to add key/value vectors for into the cache.
        cache_block_size: Number of KV vectors in each cache block.
        num_kv_heads: Number of attention kv heads.
        num_blocks: Number of blocks in the cache.
        kv_cache_dtype: KV Cache dtype.
        num_iterations: Number of iterations to record benchmark times for each impl.
        num_warmup_iterations: Number of iterations to "warmup" each impl before recording benchmark times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Triton.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
    """
    if kv_cache_dtype != "auto" and not current_platform.supports_fp8():
        error_msg = "Cannot use FP8 KV Cache because current platform does not support FP8"
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
            "num_tokens": num_tokens,
            "cache_block_size": cache_block_size,
            "num_kv_heads": num_kv_heads,
            "num_blocks": num_blocks,
            "kv_cache_dtype": kv_cache_dtype,
        },
    )

    # Create a random slot mapping.
    num_slots = cache_block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    kv = torch.randn(num_tokens, 2, num_kv_heads, head_dim, dtype=dtype)
    key, value = kv.unbind(dim=1)

    k_scale = v_scale = 2.0

    # Create the KV caches.
    key_caches_vllm, value_caches_vllm = create_kv_caches_with_random(
        num_blocks,
        cache_block_size,
        1,
        num_kv_heads,
        head_dim,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    key_cache_vllm, value_cache_vllm = key_caches_vllm[0], value_caches_vllm[0]

    key_cache, value_cache = _to_conch_layout(key_cache_vllm.clone(), value_cache_vllm.clone(), num_kv_heads, head_dim)

    # Run the reference implementation.
    reshape_and_cache_reference(
        key, value, key_cache_vllm, value_cache_vllm, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Call Triton kernel
    reshape_and_cache_triton(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)

    # Reshape vLLM key/value caches
    key_cache_vllm_out, value_cache_vllm_out = _to_conch_layout(
        key_cache_vllm, value_cache_vllm, num_kv_heads, head_dim
    )

    if not torch.allclose(key_cache, key_cache_vllm_out, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(key_cache_vllm_out - key_cache).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {key_cache}", file=sys.stderr)
            print(f"Triton output: {key_cache_vllm_out}", file=sys.stderr)
    else:
        print(f"Key cache matched with atol={absolute_tolerance} :)", file=sys.stderr)

    if not torch.allclose(value_cache, value_cache_vllm_out, atol=absolute_tolerance):
        print(f"WARNING: Reference and Triton results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(value_cache_vllm_out - value_cache).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {value_cache}", file=sys.stderr)
            print(f"Triton output: {value_cache_vllm_out}", file=sys.stderr)
    else:
        print(f"Value cache matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Benchmark Reference vs. Triton implementations
    baseline_result = benchmark_it(
        lambda: reshape_and_cache_reference(
            key,
            value,
            key_cache_vllm,
            value_cache_vllm,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        tag="Baseline",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=key_cache.device,
    )

    triton_result = benchmark_it(
        lambda: reshape_and_cache_triton(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        tag="Triton",
        metadata=metadata,
        num_iterations=num_iterations,
        num_warmup_iterations=num_warmup_iterations,
        device=key_cache.device,
    )

    # Print results
    triton_result.print_parameters(csv=csv)
    triton_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
