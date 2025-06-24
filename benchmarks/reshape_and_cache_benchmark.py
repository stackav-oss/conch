# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Conch reshape_and_cache benchmark."""

import random
import sys
from typing import Final

import click
import torch

from conch.ops.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_conch
from conch.platforms import current_platform
from conch.reference.vllm.reshape_and_cache import reshape_and_cache as reshape_and_cache_reference
from conch.third_party.vllm.utils import create_kv_cache_with_random, seed_everything
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
    "--num-tokens",
    required=True,
    type=int,
    default=512,
    help="Number of tokens",
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
    head_dim: int,
    num_tokens: int,
    cache_block_size: int,
    num_kv_heads: int,
    num_blocks: int,
    kv_cache_dtype: str,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    compile_conch: bool,
) -> None:
    """Benchmark Conch reshape_and_cache.

    Args:
        head_dim: Head dimension of input tensors.
        num_tokens: Number of tokens to add key/value vectors for into the cache.
        cache_block_size: Number of KV vectors in each cache block.
        num_kv_heads: Number of attention kv heads.
        num_blocks: Number of blocks in the cache.
        kv_cache_dtype: KV Cache dtype.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy of PyTorch vs. Conch.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag for printing results in CSV format.
        compile_ref: Flag to torch.compile() the reference implementation.
        compile_conch: Flag to torch.compile() the Conch implementation.
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

    k_scale = torch.full((1,), 2.0, dtype=torch.float32, device=device)
    v_scale = torch.full((1,), 3.0, dtype=torch.float32, device=device)

    # Create the KV caches.
    key_cache_ref, value_cache_ref = create_kv_cache_with_random(
        num_blocks,
        cache_block_size,
        num_kv_heads,
        head_dim,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )

    fp8_dtype = torch.float8_e4m3fnuz if current_platform.is_amd() else torch.float8_e4m3fn

    if "fp8" in kv_cache_dtype:
        key_cache_ref = key_cache_ref.view(fp8_dtype)
        value_cache_ref = value_cache_ref.view(fp8_dtype)

    key_cache_conch = key_cache_ref.clone()
    value_cache_conch = value_cache_ref.clone()

    reshape_and_cache_ref_fn = (
        torch.compile(reshape_and_cache_reference) if compile_ref else reshape_and_cache_reference
    )
    reshape_and_cache_conch_fn = torch.compile(reshape_and_cache_conch) if compile_conch else reshape_and_cache_conch

    # Run the reference implementation.
    reshape_and_cache_ref_fn(key, value, key_cache_ref, value_cache_ref, slot_mapping, kv_cache_dtype, k_scale, v_scale)

    # Call Conch kernel
    reshape_and_cache_conch_fn(
        key, value, key_cache_conch, value_cache_conch, slot_mapping, kv_cache_dtype, k_scale, v_scale
    )

    # Can't compare FP8 directly, so bitcast to uint8 for comparison
    if "fp8" in kv_cache_dtype:
        key_cache_ref = key_cache_ref.view(torch.uint8)
        value_cache_ref = value_cache_ref.view(torch.uint8)
        key_cache_conch = key_cache_conch.view(torch.uint8)
        value_cache_conch = value_cache_conch.view(torch.uint8)

    if not torch.allclose(key_cache_conch, key_cache_ref, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(key_cache_ref - key_cache_conch).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {key_cache_conch}", file=sys.stderr)
            print(f"Conch output: {key_cache_ref}", file=sys.stderr)
    else:
        print(f"Key cache matched with atol={absolute_tolerance} :)", file=sys.stderr)

    if not torch.allclose(value_cache_conch, value_cache_ref, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(value_cache_ref - value_cache_conch).abs().max().item()}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {value_cache_conch}", file=sys.stderr)
            print(f"Conch output: {value_cache_ref}", file=sys.stderr)
    else:
        print(f"Value cache matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Convert datatype back to FP8 before benchmark
    if "fp8" in kv_cache_dtype:
        key_cache_ref = key_cache_ref.view(fp8_dtype)
        value_cache_ref = value_cache_ref.view(fp8_dtype)
        key_cache_conch = key_cache_conch.view(fp8_dtype)
        value_cache_conch = value_cache_conch.view(fp8_dtype)

    # Benchmark Reference vs. Conch implementations
    baseline_result = benchmark_it(
        lambda: reshape_and_cache_ref_fn(
            key,
            value,
            key_cache_ref,
            value_cache_ref,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: reshape_and_cache_conch_fn(
            key,
            value,
            key_cache_conch,
            value_cache_conch,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
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
