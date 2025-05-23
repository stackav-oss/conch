# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test Triton bnb quantization implementation."""

from typing import Final

import pytest
import torch
import triton
import triton.language as tl

from conch.kernels.quantization.bitsandbytes.quantize_blockwise import _fp4_quantize, _nf4_quantize
from conch.kernels.quantization.bitsandbytes.quantize_blockwise import (
    quantize_blockwise_launcher as quantize_blockwise_triton,
)
from conch.ops.quantization.bitsandbytes.functional import (
    SUPPORTED_BLOCKSIZES,
    SUPPORTED_QUANT_TYPES,
    _create_dynamic_map,
    get_absmax_shape,
    get_quantized_output_shape,
)
from conch.platforms import current_platform
from conch.reference.quantization.bitsandbytes.quantize_blockwise import _pytorch_fp4_quantize, _pytorch_nf4_quantize
from conch.reference.quantization.bitsandbytes.quantize_blockwise import (
    quantize_blockwise_launcher as quantize_blockwise_reference,
)
from conch.third_party.vllm.utils import seed_everything

# Too many parameterizations makes the PyTorch-reference test cases too slow
_BLOCKSIZES_ABRIDGED: Final = [64, 1024]
_SIZE_MULTIPLIERS: Final = [2.5, 6]
_DTYPES: Final = [torch.float32, torch.float16, torch.bfloat16]


@triton.jit  # type: ignore[misc]
def _quantize_fp4_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor) -> None:
    val = tl.load(x_ptr)
    q_val = _fp4_quantize(val)
    tl.store(out_ptr, q_val)


@triton.jit  # type: ignore[misc]
def _quantize_nf4_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor) -> None:
    val = tl.load(x_ptr)
    q_val = _nf4_quantize(val)
    tl.store(out_ptr, q_val)


def _quantize_launcher(val: float, quant_type: str) -> int:
    device = torch.device(current_platform.device)

    x = torch.full((1,), val, dtype=torch.float32, device=device)
    out = torch.empty_like(x, dtype=torch.uint8, device=device)

    quantize_kernel = _quantize_fp4_kernel if quant_type == "fp4" else _quantize_nf4_kernel
    quantize_kernel[(1,)](x, out)

    return int(out.cpu()[0].item())


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (1.00, 15),
        (0.87, 15),
        (0.85, 14),
        (0.65, 14),
        (0.63, 13),
        (0.51, 13),
        (0.49, 12),
        (0.39, 12),
        (0.38, 11),
        (0.30, 11),
        (0.29, 10),
        (0.21, 10),
        (0.20, 9),
        (0.13, 9),
        (0.12, 8),
        (0.04, 8),
        (0.03, 7),
        (-0.04, 7),
        (-0.05, 6),
        (-0.13, 6),
        (-0.14, 5),
        (-0.23, 5),
        (-0.25, 4),
        (-0.33, 4),
        (-0.34, 3),
        (-0.45, 3),
        (-0.46, 2),
        (-0.60, 2),
        (-0.62, 1),
        (-0.84, 1),
        (-0.85, 0),
        (-1.0, 0),
    ],
)
def test_quantize_nf4(x: float, expected: int) -> None:
    """Test NF4 mapping."""
    quant_type: Final = "nf4"
    assert _quantize_launcher(x, quant_type) == expected
    assert _pytorch_nf4_quantize(torch.tensor(x), None) == expected


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (1.0, 3),
        (0.84, 3),
        (0.83, 2),
        (0.59, 2),
        (0.58, 5),
        (0.42, 5),
        (0.41, 4),
        (0.30, 4),
        (0.29, 7),
        (0.21, 7),
        (0.20, 6),
        (0.09, 6),
        (0.08, 1),
        (0.003, 1),
        (0.002, 0),
        (0.0, 0),
        (-0.000001, 8),
        (-0.002, 8),
        (-0.003, 9),
        (-0.08, 9),
        (-0.09, 14),
        (-0.20, 14),
        (-0.21, 15),
        (-0.29, 15),
        (-0.30, 12),
        (-0.41, 12),
        (-0.42, 13),
        (-0.58, 13),
        (-0.59, 10),
        (-0.83, 10),
        (-0.84, 11),
        (-1.0, 11),
    ],
)
def test_quantize_fp4(x: float, expected: int) -> None:
    """Test FP4 mapping."""
    quant_type: Final = "fp4"
    assert _quantize_launcher(x, quant_type) == expected
    assert _pytorch_fp4_quantize(torch.tensor(x), None) == expected


@pytest.mark.parametrize("blocksize", _BLOCKSIZES_ABRIDGED)
@pytest.mark.parametrize("size_multiplier", _SIZE_MULTIPLIERS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("quant_type", SUPPORTED_QUANT_TYPES)
def test_quantize_blockwise(blocksize: int, size_multiplier: float, dtype: torch.dtype, quant_type: str) -> None:
    """Test blockwise quantization method."""
    assert blocksize in SUPPORTED_BLOCKSIZES

    # TODO(jmanning):
    # Seeing odd AMD compiler error for this kernel for FP8 cases
    # E       RuntimeError: PassManager::run failed
    # .direnv/python-3.10.12/lib/python3.10/site-packages/triton/backends/amd/compiler.py:243: RuntimeError
    # loc("/home/$USER/conch/conch/kernels/quantization/bitsandbytes/quantize_blockwise.py":114:21): error: operand #1 does not dominate this use
    if current_platform.is_amd() and quant_type == "fp8":
        pytest.skip()

    # There are some small rounding discrepancies between Triton and PyTorch, and its tough to adjust tolerances
    # to account for it because we compre bitpacked results (so if there's a rounding discrepancy in what become
    # the higher-bits of the result, the tolerance would need to be much higher to account for it). Consider changing
    # this seed if test cases are failing
    seed: Final = 2
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    torch.set_default_device(device)

    input_size = int(blocksize * size_multiplier)
    output_shape: Final = get_quantized_output_shape(input_size, quant_type)
    absmax_shape: Final = get_absmax_shape(input_size, blocksize)

    x = torch.randn((input_size,), device=device, dtype=dtype)
    if quant_type == "fp4":
        x = x.uniform_(-1.0, 1.0)

    reference_absmax = torch.empty(absmax_shape, device=device, dtype=dtype)
    reference_out = torch.empty(output_shape, device=device, dtype=torch.uint8)

    code = _create_dynamic_map() if quant_type == "fp8" else None

    quantize_blockwise_reference(
        x,
        reference_absmax,
        reference_out,
        code,
        blocksize,
        input_size,
        quant_type,
    )

    triton_absmax = torch.empty(absmax_shape, device=device, dtype=dtype)
    triton_out = torch.empty(output_shape, device=device, dtype=torch.uint8)

    quantize_blockwise_triton(
        x,
        triton_absmax,
        triton_out,
        code,
        blocksize,
        input_size,
        quant_type,
    )

    torch.testing.assert_close(reference_absmax, triton_absmax)
    torch.testing.assert_close(reference_out, triton_out, atol=1, rtol=5e-3)
