# SPDX-License-Identifier: Apache-2.0

import random
from typing import Final

import torch

from conch import envs
from conch.platforms import current_platform

STR_DTYPE_TO_TORCH_DTYPE: Final = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _generate_random_fp8(
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    # -----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    if envs.CONCH_ENABLE_VLLM and current_platform.has_cuda():
        from vllm import _custom_ops as ops

        ops.convert_fp8(tensor, tensor_tmp)
    else:
        from conch.reference.quantization.fp8 import _scaled_fp8_quant_pytorch_ref

        tensor = _scaled_fp8_quant_pytorch_ref(tensor_tmp, scale=torch.tensor(1.0))
    del tensor_tmp


def get_kv_cache_torch_dtype(
    cache_dtype: str | torch.dtype | None, model_dtype: str | torch.dtype | None = None
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                error_msg = f"Invalid model dtype: {model_dtype}"
                raise ValueError(error_msg)
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            error_msg = f"Invalid kv cache dtype: {cache_dtype}"
            raise ValueError(error_msg)
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        error_msg = f"Invalid kv cache dtype: {cache_dtype}"
        raise TypeError(error_msg)
    return torch_dtype


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        error_msg = f"Does not support key cache of type fp8 with head_size {head_size}"
        raise ValueError(error_msg)

    seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            error_msg = f"Does not support key cache of type {cache_dtype}"
            raise ValueError(error_msg)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            error_msg = f"Does not support value cache of type {cache_dtype}"
            raise ValueError(error_msg)
        value_caches.append(value_cache)
    return key_caches, value_caches


def reshape_vllm_kvcache(
    key_cache_vllm: torch.Tensor, value_cache_vllm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape vLLM key and value caches into format expected by FlashAttention.

    Args:
        key_cache_vllm: vLLM key cache, shape: (num_cache_blocks, num_kv_heads, head_size // x, cache_block_size, x).
        value_cache_vllm: vLLM value cache, shape: (num_cache_blocks, num_kv_heads, head_size, cache_block_size).

    Returns:
        Reshaped key and value caches as (num_cache_blocks, cache_block_size, num_kv_heads, head_size).
    """
    num_cache_blocks, num_kv_heads, head_size, cache_block_size = value_cache_vllm.shape

    k = key_cache_vllm.permute(0, 1, 3, 2, 4).contiguous().reshape(num_cache_blocks, num_kv_heads, cache_block_size, head_size)
    v = value_cache_vllm.permute(0, 1, 3, 2).contiguous().reshape(num_cache_blocks, num_kv_heads, cache_block_size, head_size)

    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    return k, v


def create_tensors(
    head_dim: int,
    seq_len: int,
    cache_block_size: int,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_cache_dtype: str,
    gpu: str,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V, block tables, and sequence lengths tensors for benchmarking."""
    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    assert num_query_heads % num_kv_heads == 0

    scale: Final = float(1.0 / (head_dim**0.5))
    total_num_cache_elements: Final = batch_size * seq_len
    num_cache_blocks: Final = total_num_cache_elements // cache_block_size

    # Make input tensors
    query = torch.empty(batch_size, num_query_heads, head_dim, dtype=dtype)
    query.uniform_(-scale, scale)

    seq_lens_lst = [random.randint(1, seq_len) for _ in range(batch_size)]  # noqa: S311
    seq_lens_lst[-1] = seq_len
    max_seq_len = max(seq_lens_lst)
    seq_lens = torch.as_tensor(seq_lens_lst, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + cache_block_size - 1) // cache_block_size
    block_table_lst: list[list[int]] = []
    for _ in range(batch_size):
        block_table = [random.randint(0, num_cache_blocks - 1) for _ in range(max_num_blocks_per_seq)]  # noqa: S311
        block_table_lst.append(block_table)

    block_table = torch.as_tensor(block_table_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches_vllm, value_caches_vllm = create_kv_caches_with_random(
        num_cache_blocks,
        cache_block_size,
        1,
        num_kv_heads,
        head_dim,
        kv_cache_dtype,
        dtype,
        seed,
        device,  # type: ignore[arg-type]
    )

    key_cache_vllm, value_cache_vllm = key_caches_vllm[0], value_caches_vllm[0]

    key_cache_flash, value_cache_flash = reshape_vllm_kvcache(key_cache_vllm, value_cache_vllm)

    return query, key_cache_vllm, value_cache_vllm, key_cache_flash, value_cache_flash, block_table, seq_lens
