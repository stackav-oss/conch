# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test Triton bnb dequantization implementation."""

from typing import Final

import pytest
import torch
import triton
import triton.language as tl

from conch.kernels.quantization.bitsandbytes.dequantize_blockwise import _fp4_dequantize, _nf4_dequantize
from conch.kernels.quantization.bitsandbytes.dequantize_blockwise import (
    dequantize_blockwise_launcher as dequantize_blockwise_triton,
)
from conch.ops.quantization.bitsandbytes.functional import (
    SUPPORTED_BLOCKSIZES,
    SUPPORTED_QUANT_TYPES,
    _create_dynamic_map,
)
from conch.platforms import current_platform
from conch.reference.quantization.bitsandbytes.dequantize_blockwise import (
    _pytorch_fp4_dequantize,
    _pytorch_nf4_dequantize,
)
from conch.reference.quantization.bitsandbytes.dequantize_blockwise import (
    dequantize_blockwise_launcher as dequantize_blockwise_reference,
)
from conch.third_party.vllm.utils import seed_everything

# Too many parameterizations makes the PyTorch-reference test cases too slow
_BLOCKSIZES_ABRIDGED: Final = [64, 1024]
_SIZE_MULTIPLIERS: Final = [2.5, 6]
_DTYPES: Final = [torch.float32, torch.float16, torch.bfloat16]


@triton.jit  # type: ignore[misc]
def _dequantize_fp4_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor) -> None:
    val = tl.load(x_ptr)
    q_val = _fp4_dequantize(val)
    tl.store(out_ptr, q_val)


@triton.jit  # type: ignore[misc]
def _dequantize_nf4_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor) -> None:
    val = tl.load(x_ptr)
    q_val = _nf4_dequantize(val)
    tl.store(out_ptr, q_val)


def _dequantize_launcher(val: int, quant_type: str) -> float:
    device = torch.device(current_platform.device)

    x = torch.full((1,), val, dtype=torch.uint8, device=device)
    out = torch.empty_like(x, dtype=torch.float32, device=device)

    dequantize_kernel = _dequantize_fp4_kernel if quant_type == "fp4" else _dequantize_nf4_kernel
    dequantize_kernel[(1,)](x, out)

    return out.cpu()[0].item()


@pytest.mark.parametrize(
    ("expected", "x"),
    [
        (-0.25, 15),
        (-0.166666, 14),
        (-0.5, 13),
        (-0.333333, 12),
        (-1.0, 11),
        (-0.6666666, 10),
        (-0.0052083333, 9),
        (-0.0, 8),
        (0.25, 7),
        (0.166666, 6),
        (0.5, 5),
        (0.333333, 4),
        (1.0, 3),
        (0.6666666, 2),
        (0.0052083333, 1),
        (0.0, 0),
    ],
)
def test_dequantize_fp4(expected: float, x: int) -> None:
    """Test FP4 mapping."""
    quant_type: Final = "fp4"
    assert _dequantize_launcher(x, quant_type) == pytest.approx(expected)
    assert _pytorch_fp4_dequantize(torch.tensor(x), None) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("expected", "x"),
    [
        (1.00, 15),
        (0.7229568362236023, 14),
        (0.5626170039176941, 13),
        (0.44070982933044434, 12),
        (0.33791524171829224, 11),
        (0.24611230194568634, 10),
        (0.16093020141124725, 9),
        (0.07958029955625534, 8),
        (0.0, 7),
        (-0.09105003625154495, 6),
        (-0.18477343022823334, 5),
        (-0.28444138169288635, 4),
        (-0.39491748809814453, 3),
        (-0.5250730514526367, 2),
        (-0.6961928009986877, 1),
        (-1.0, 0),
    ],
)
def test_dequantize_nf4(expected: float, x: int) -> None:
    """Test NF4 mapping."""
    quant_type: Final = "nf4"
    assert _dequantize_launcher(x, quant_type) == pytest.approx(expected)
    assert _pytorch_nf4_dequantize(torch.tensor(x), None) == pytest.approx(expected)


@pytest.mark.parametrize("blocksize", _BLOCKSIZES_ABRIDGED)
@pytest.mark.parametrize("size_multiplier", _SIZE_MULTIPLIERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("quant_type", SUPPORTED_QUANT_TYPES)
def test_dequantize(blocksize: int, size_multiplier: float, dtype: torch.dtype, quant_type: str) -> None:
    """Test blockwise dequantization method."""
    assert blocksize in SUPPORTED_BLOCKSIZES

    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    output_size = int(blocksize * size_multiplier)
    num_blocks = triton.cdiv(output_size, blocksize)
    input_size = output_size if quant_type == "fp8" else output_size // 2

    x = torch.randint(0, 255, (input_size,), device=device, dtype=torch.uint8)
    absmax = torch.randn((num_blocks,), device=device, dtype=dtype)
    triton_out = torch.empty((output_size,), device=device, dtype=dtype)
    reference_out = torch.empty((output_size,), device=device, dtype=dtype)

    code = _create_dynamic_map() if quant_type == "fp8" else None

    triton_result = dequantize_blockwise_triton(
        x,
        absmax=absmax,
        out=triton_out,
        code=code,
        blocksize=blocksize,
        output_size=output_size,
        quant_type=quant_type,
    )

    reference_result = dequantize_blockwise_reference(
        x,
        absmax=absmax,
        out=reference_out,
        code=code,
        blocksize=blocksize,
        output_size=output_size,
        quant_type=quant_type,
    )

    torch.testing.assert_close(triton_result, reference_result)
