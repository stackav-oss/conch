# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""GEMM kernel supporting mixed-precision (e.g. w{1|2|4|8|16}, a{16|32}) and scaled matrix multiplications (fp8/int8 with scaling factors to different output dtype).

Heavily based on [GemLite](https://github.com/mobiusml/gemlite).
Core algorithm is the same, but we add a new frontend for various use-cases (mixed-precision and scaled).
"""

from dataclasses import dataclass
from enum import Enum

import torch
import triton
import triton.language as tl

from conch.platforms import current_platform
from conch.third_party.vllm.scalar_type import ScalarType


class DType(Enum):
    """Data types for mapping between Triton/Torch."""

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8 = 3
    INT8 = 4
    UINT8 = 5
    INT32 = 6
    UINT32 = 7
    FP8E5 = 8


class WeightGroupMode(Enum):
    """Dequantization weight group modes."""

    NONE = 0
    SHIFT = 1
    SYMMETRIC_NO_SHIFT = 2
    SYMMETRIC_WITH_SHIFT = 3
    ASYMMETRIC = 4


# Triton can only access global values instantiated as tl.constexpr
_WEIGHT_GROUP_MODE_NONE: tl.constexpr = tl.constexpr(WeightGroupMode.NONE.value)
_WEIGHT_GROUP_MODE_SHIFT: tl.constexpr = tl.constexpr(WeightGroupMode.SHIFT.value)
_WEIGHT_GROUP_MODE_SYMMETRIC_NO_SHIFT: tl.constexpr = tl.constexpr(WeightGroupMode.SYMMETRIC_NO_SHIFT.value)
_WEIGHT_GROUP_MODE_SYMMETRIC_WITH_SHIFT: tl.constexpr = tl.constexpr(WeightGroupMode.SYMMETRIC_WITH_SHIFT.value)
_WEIGHT_GROUP_MODE_ASYMMETRIC: tl.constexpr = tl.constexpr(WeightGroupMode.ASYMMETRIC.value)


class LoadOrder(Enum):
    """Order to load A matrix relative to other tensors."""

    VERY_EARLY = 0
    EARLY = 1
    MID = 2
    LATE = 3


# Triton can only access global values instantiated as tl.constexpr
_LOAD_ORDER_VERY_EARLY: tl.constexpr = tl.constexpr(LoadOrder.VERY_EARLY.value)
_LOAD_ORDER_EARLY: tl.constexpr = tl.constexpr(LoadOrder.EARLY.value)
_LOAD_ORDER_MID: tl.constexpr = tl.constexpr(LoadOrder.MID.value)
_LOAD_ORDER_LATE: tl.constexpr = tl.constexpr(LoadOrder.LATE.value)


class ChannelScaleMode(Enum):
    """Mode for channel scaling."""

    NONE = 0
    WEIGHT_ONLY = 1
    ACTIVATION_ONLY = 2
    WEIGHT_AND_ACTIVATION = 3


# Triton can only access global values instantiated as tl.constexpr
_CHANNEL_SCALE_MODE_NONE: tl.constexpr = tl.constexpr(ChannelScaleMode.NONE.value)
_CHANNEL_SCALE_MODE_WEIGHT_ONLY: tl.constexpr = tl.constexpr(ChannelScaleMode.WEIGHT_ONLY.value)
_CHANNEL_SCALE_MODE_ACTIVATION_ONLY: tl.constexpr = tl.constexpr(ChannelScaleMode.ACTIVATION_ONLY.value)
_CHANNEL_SCALE_MODE_WEIGHT_AND_ACTIVATION: tl.constexpr = tl.constexpr(ChannelScaleMode.WEIGHT_AND_ACTIVATION.value)


DTYPE_TO_TORCH = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.float8_e4m3fn,
    4: torch.int8,
    5: torch.uint8,
    6: torch.int32,
    7: torch.uint32,
    8: torch.float8_e5m2,
    9: torch.float8_e4m3fnuz,
}

TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.uint16: tl.uint16,
    torch.int32: tl.int32,
    torch.uint32: tl.uint32,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e4m3fnuz: tl.float8e4b8,
}

DTYPE_TO_TRITON = {key: TORCH_DTYPE_TO_TRITON[dtype] for key, dtype in DTYPE_TO_TORCH.items()}


def _get_matrix_a_eviction_policy() -> str:
    """Get eviction policy for Matrix A based on platform."""
    if current_platform.is_nvidia():
        return "evict_last"

    return ""


def _get_matrix_b_eviction_policy() -> str:
    """Get eviction policy for Matrix B based on platform."""
    if current_platform.is_nvidia():
        return "evict_first"

    return ""


def _get_metadata_eviction_policy() -> str:
    """Get eviction policy for metadata (scales and zeros) based on platform."""
    return ""


def _get_tuning_parameters() -> dict[str, int]:
    """Get block sizes/tuning parameters for current device."""
    device_name = current_platform.get_device_name()

    if "H100" in device_name:
        return {
            "cxpr_block_size_m": 128,
            "cxpr_block_size_n": 128,
            "cxpr_block_size_k": 128,
            "cxpr_group_size_m": 8,
            "num_warps": 8,
            "num_stages": 2,
        }

    if "MI300X" in device_name:
        return {
            "cxpr_block_size_m": 128,
            "cxpr_block_size_n": 64,
            "cxpr_block_size_k": 128,
            "cxpr_group_size_m": 16,
            "num_warps": 8,
            "num_stages": 2,
        }

    return {
        "cxpr_block_size_m": 64,
        "cxpr_block_size_n": 64,
        "cxpr_block_size_k": 32,
        "cxpr_group_size_m": 8,
    }


@triton.jit  # type: ignore[misc]
def _swizzle_tile(
    pid: int,
    m_dim: int,
    n_dim: int,
    cxpr_block_size_m: tl.constexpr,
    cxpr_block_size_n: tl.constexpr,
    cxpr_group_size_m: tl.constexpr,
) -> tuple[int, int]:
    """Return pid based on a swizzled tile of size (m, n)."""
    grid_m = tl.cdiv(m_dim, cxpr_block_size_m)
    grid_n = tl.cdiv(n_dim, cxpr_block_size_n)
    width = cxpr_group_size_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * cxpr_group_size_m, cxpr_group_size_m)
    pid_m = group_id * cxpr_group_size_m + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@triton.jit  # type: ignore[misc]
def _linear_tile(pid: int, n_dim: int, cxpr_block_size_n: tl.constexpr) -> tuple[int, int]:
    """Return pid based on a linear tile of size (m, n)."""
    pid_m = pid // tl.cdiv(n_dim, cxpr_block_size_n)
    pid_n = pid % tl.cdiv(n_dim, cxpr_block_size_n)
    return pid_m, pid_n


@triton.jit  # type: ignore[misc]
def _dequantize(
    b: tl.tensor,
    scales: tl.tensor,
    zeros: tl.tensor,
    q_shift: tl.tensor,
    cxpr_input_dtype: tl.constexpr,
    cxpr_meta_dtype: tl.constexpr,
    cxpr_unpack_mask: tl.constexpr,
    cxpr_elements_per_sample: tl.constexpr,
    cxpr_w_group_mode: tl.constexpr,
    cxpr_zero_is_scalar: tl.constexpr,
    cxpr_weight_bias: tl.constexpr,
) -> tl.tensor:
    """Dequantize a tensor."""
    # Unpack
    if cxpr_elements_per_sample > 1:
        b = ((b >> q_shift) & cxpr_unpack_mask).to(cxpr_meta_dtype)
        b -= cxpr_weight_bias

    # Shift (operation: b - zeros)
    if cxpr_w_group_mode == _WEIGHT_GROUP_MODE_SHIFT:
        b -= zeros

    # Symmetric no shift (operation: b * scales)
    if cxpr_w_group_mode == _WEIGHT_GROUP_MODE_SYMMETRIC_NO_SHIFT:
        b = b.to(cxpr_meta_dtype) * scales

    # Asymmetric / Symmetric with shift (operation: (b - zeros) * scales)
    if cxpr_w_group_mode == _WEIGHT_GROUP_MODE_SYMMETRIC_WITH_SHIFT:
        b = (
            (b - zeros).to(cxpr_meta_dtype) * scales
            if cxpr_zero_is_scalar
            else (b.to(cxpr_meta_dtype) - zeros) * scales
        )

    # Asymmetric (operation: b * scales + zeros)
    if cxpr_w_group_mode == _WEIGHT_GROUP_MODE_ASYMMETRIC:
        b = tl.fma(b.to(cxpr_meta_dtype), scales, zeros)

    return b.to(cxpr_input_dtype)


@triton.jit  # type: ignore[misc]
def _gemm_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    scales_ptr: tl.tensor,
    zeros_ptr: tl.tensor,
    scales_a_ptr: tl.tensor,
    # Input sizes
    m_dim: int,
    k_dim: int,
    n_dim: int,
    # Quantization parameters
    cxpr_w_nbits: tl.constexpr,
    cxpr_weight_bias: tl.constexpr,
    cxpr_group_size: tl.constexpr,
    cxpr_unpack_mask: tl.constexpr,
    cxpr_elements_per_sample: tl.constexpr,
    # Tensor strides
    matrix_a_stride_m: int,
    matrix_a_stride_k: int,
    matrix_b_stride_k: int,
    matrix_b_stride_n: int,
    matrix_c_stride_m: int,
    matrix_c_stride_n: int,
    meta_stride_g: int,
    meta_stride_n: int,
    # Data types
    cxpr_input_dtype: tl.constexpr,
    cxpr_output_dtype: tl.constexpr,
    cxpr_acc_dtype: tl.constexpr,
    cxpr_meta_dtype: tl.constexpr,
    # Metadata modes
    cxpr_channel_scale_mode: tl.constexpr,
    cxpr_w_group_mode: tl.constexpr,
    cxpr_zero_is_scalar: tl.constexpr,
    # Tuning parameters
    cxpr_block_size_m: tl.constexpr,
    cxpr_block_size_n: tl.constexpr,
    cxpr_block_size_k: tl.constexpr,
    cxpr_group_size_m: tl.constexpr,
    cxpr_block_size_scale_a: tl.constexpr,
    cxpr_block_size_scale_b: tl.constexpr,
    cxpr_matrix_a_load_order: tl.constexpr,
    cxpr_matrix_a_eviction_policy: tl.constexpr,
    cxpr_matrix_b_eviction_policy: tl.constexpr,
    cxpr_meta_eviction_policy: tl.constexpr,
    cxpr_data_contiguous: tl.constexpr,
    cxpr_swizzle_pid: tl.constexpr,
) -> None:
    """Based on https://github.com/fpgaminer/GPTQ-triton.

    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (m_dim, k_dim): float16 or bfloat16
    B is of shape (k_dim//elements_per_sample, n_dim): int32 as a packed matrix
    C is of shape (m_dim, n_dim): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (k_dim // group_size, n_dim): float16 or bfloat16
    cxpr_block_size_m >=16
    cxpr_block_size_k <= group_size
    """
    pid = tl.program_id(axis=0)

    pid_m, pid_n = (
        _swizzle_tile(pid, m_dim, n_dim, cxpr_block_size_m, cxpr_block_size_n, cxpr_group_size_m)
        if cxpr_swizzle_pid
        else _linear_tile(pid, n_dim, cxpr_block_size_n)
    )

    num_groups = tl.cdiv(k_dim, cxpr_block_size_k)

    matrix_a_offsets_m = pid_m * cxpr_block_size_m + tl.arange(0, cxpr_block_size_m)
    scale_a_offsets_m = (
        pid_m * cxpr_block_size_scale_a + tl.arange(0, cxpr_block_size_scale_a)
        if cxpr_block_size_scale_a > 1
        else tl.arange(0, 1)
    )

    matrix_b_offsets_n = pid_n * cxpr_block_size_n + tl.arange(0, cxpr_block_size_n)
    scale_b_offsets_n = (
        pid_n * cxpr_block_size_scale_b + tl.arange(0, cxpr_block_size_scale_b)
        if cxpr_block_size_scale_b > 1
        else tl.arange(0, 1)
    )

    group_offsets = tl.arange(0, cxpr_block_size_k)

    # Vectorized coalesced load
    if cxpr_data_contiguous:
        matrix_b_offsets_n = tl.max_contiguous(tl.multiple_of(matrix_b_offsets_n, cxpr_block_size_n), cxpr_block_size_n)

    # Calculate offsets for pointer into the block of matrix A
    matrix_a_block_offsets = (
        matrix_a_offsets_m[:, None] * matrix_a_stride_m + group_offsets[None, :] * matrix_a_stride_k
    )
    # Create pointer to block of matrix A
    matrix_a_block_ptr = a_ptr + matrix_a_block_offsets
    # Mask out any out-of-bounds elements
    matrix_a_block_mask = matrix_a_offsets_m[:, None] < m_dim

    # Calculate offsets for pointer into the block of matrix B
    matrix_b_block_offsets = (
        group_offsets[:, None] // cxpr_elements_per_sample
    ) * matrix_b_stride_k + matrix_b_offsets_n[None, :] * matrix_b_stride_n
    # Create pointer to block of matrix B
    matrix_b_block_ptr = b_ptr + matrix_b_block_offsets

    # Quantization metadata
    q_shift = ((group_offsets % cxpr_elements_per_sample) * cxpr_w_nbits).to(tl.int32)[:, None]
    scales_block_ptr = scales_ptr + matrix_b_offsets_n[None, :] * meta_stride_n
    zeros_block_ptr = zeros_ptr + matrix_b_offsets_n[None, :] * meta_stride_n
    stride_mul = cxpr_block_size_k / cxpr_group_size

    if cxpr_zero_is_scalar:
        zero_scalar = tl.load(zeros_ptr, eviction_policy=cxpr_matrix_a_eviction_policy)

    # Output accumulator
    acc = tl.zeros((cxpr_block_size_m, cxpr_block_size_n), dtype=cxpr_acc_dtype)

    for group_index in range(num_groups):
        # Very early load
        if cxpr_matrix_a_load_order == _LOAD_ORDER_VERY_EARLY:
            a = tl.load(
                matrix_a_block_ptr, mask=matrix_a_block_mask, other=0.0, eviction_policy=cxpr_matrix_a_eviction_policy
            )

        b = tl.load(matrix_b_block_ptr, eviction_policy=cxpr_matrix_b_eviction_policy)

        # Early load
        if cxpr_matrix_a_load_order == _LOAD_ORDER_EARLY:
            a = tl.load(
                matrix_a_block_ptr, mask=matrix_a_block_mask, other=0.0, eviction_policy=cxpr_matrix_a_eviction_policy
            )

        # Load meta-data
        if cxpr_w_group_mode > _WEIGHT_GROUP_MODE_NONE:
            group_offset = (group_index * stride_mul).to(tl.int32) * meta_stride_g

        # Load scales if they exist
        scales = None
        if cxpr_w_group_mode >= _WEIGHT_GROUP_MODE_SYMMETRIC_NO_SHIFT:
            scales = tl.load(scales_block_ptr + group_offset, eviction_policy=cxpr_meta_eviction_policy)

        # Load zeros if they exist
        zeros = None
        if (
            cxpr_w_group_mode == _WEIGHT_GROUP_MODE_SHIFT
            or cxpr_w_group_mode >= _WEIGHT_GROUP_MODE_SYMMETRIC_WITH_SHIFT
        ):
            zeros = (
                zero_scalar
                if cxpr_zero_is_scalar
                else tl.load(zeros_block_ptr + group_offset, eviction_policy=cxpr_meta_eviction_policy)
            ).to(cxpr_meta_dtype)

        # Mid load
        if cxpr_matrix_a_load_order == _LOAD_ORDER_MID:
            a = tl.load(
                matrix_a_block_ptr, mask=matrix_a_block_mask, other=0.0, eviction_policy=cxpr_matrix_a_eviction_policy
            )

        # Unpack and dequantize
        b = _dequantize(
            b,
            scales,
            zeros,
            q_shift,
            cxpr_input_dtype,
            cxpr_meta_dtype,
            cxpr_unpack_mask,
            cxpr_elements_per_sample,
            cxpr_w_group_mode,
            cxpr_zero_is_scalar,
            cxpr_weight_bias,
        )

        # Late load
        if cxpr_matrix_a_load_order == _LOAD_ORDER_LATE:
            a = tl.load(
                matrix_a_block_ptr, mask=matrix_a_block_mask, other=0.0, eviction_policy=cxpr_matrix_a_eviction_policy
            )

        # Dot
        acc = tl.dot(a, b, acc=acc, out_dtype=cxpr_acc_dtype)

        # Advance
        matrix_a_block_ptr += cxpr_block_size_k * matrix_a_stride_k
        matrix_b_block_ptr += (cxpr_block_size_k // cxpr_elements_per_sample) * matrix_b_stride_k

    # Channel-wise scaling
    if cxpr_channel_scale_mode == _CHANNEL_SCALE_MODE_WEIGHT_ONLY:
        scales_b = tl.load(
            scales_ptr + matrix_b_offsets_n,
            mask=matrix_b_offsets_n < n_dim,
            other=1,
            eviction_policy=cxpr_meta_eviction_policy,
        )
        acc = acc.to(cxpr_meta_dtype) * scales_b[None, :]

    if cxpr_channel_scale_mode == _CHANNEL_SCALE_MODE_ACTIVATION_ONLY:
        scales_a = tl.load(
            scales_a_ptr + matrix_a_offsets_m,
            mask=matrix_a_offsets_m < m_dim,
            other=1,
            eviction_policy=cxpr_meta_eviction_policy,
        )
        scales_b = tl.full((cxpr_block_size_n,), value=1, dtype=cxpr_meta_dtype)
        acc = acc.to(cxpr_meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if cxpr_channel_scale_mode == _CHANNEL_SCALE_MODE_WEIGHT_AND_ACTIVATION:
        scales_a = tl.load(
            scales_a_ptr + scale_a_offsets_m,
            mask=scale_a_offsets_m < m_dim,
            other=1,
            eviction_policy=cxpr_meta_eviction_policy,
        )
        scales_b = tl.load(
            scales_ptr + scale_b_offsets_n,
            mask=scale_b_offsets_n < n_dim,
            other=1,
            eviction_policy=cxpr_meta_eviction_policy,
        )
        acc = acc.to(cxpr_meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    # Calculate offsets for pointer into the block of the output matrix
    matrix_c_block_offsets_m = pid_m * cxpr_block_size_m + tl.arange(0, cxpr_block_size_m)
    matrix_c_block_offsets_n = pid_n * cxpr_block_size_n + tl.arange(0, cxpr_block_size_n)
    matrix_c_block_offsets_n = tl.max_contiguous(
        tl.multiple_of(matrix_c_block_offsets_n, cxpr_block_size_n), cxpr_block_size_n
    )

    # Create pointer to block of output matrix to store accumulated result
    matrix_c_block_ptr = c_ptr + (
        matrix_c_block_offsets_m[:, None] * matrix_c_stride_m + matrix_c_block_offsets_n[None, :] * matrix_c_stride_n
    )
    # Create mask of elements to store
    matrix_c_block_mask = (matrix_c_block_offsets_m[:, None] < m_dim) & (matrix_c_block_offsets_n[None, :] < n_dim)

    # Store accumulated result
    tl.store(matrix_c_block_ptr, acc.to(cxpr_output_dtype), mask=matrix_c_block_mask)


@dataclass
class MixedPrecisionMatmulMetadata:
    """Metadata for GEMM kernel."""

    m_dim: int
    k_dim: int
    n_dim: int
    weight_type: ScalarType
    weight_bias: int
    group_size: int
    elements_per_sample: int
    zero_is_scalar: bool
    unpack_mask: int
    data_contiguous: bool
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    acc_dtype: torch.dtype
    meta_dtype: torch.dtype
    channel_scale_mode: ChannelScaleMode
    weight_group_mode: WeightGroupMode


def mixed_precision_gemm_launcher(
    output: torch.Tensor,
    x: torch.Tensor,
    w_q_packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    metadata: MixedPrecisionMatmulMetadata,
) -> None:
    """GEMM Triton Launcher."""
    grid = lambda parameters: (  # noqa: E731
        triton.cdiv(metadata.m_dim, parameters["cxpr_block_size_m"])
        * triton.cdiv(metadata.n_dim, parameters["cxpr_block_size_n"]),
    )

    zeros = zeros if zeros is not None else torch.tensor([[]], dtype=torch.int32, device=x.device)

    tuning_parameters = _get_tuning_parameters()

    _gemm_kernel[grid](
        a_ptr=x,
        b_ptr=w_q_packed,
        c_ptr=output,
        scales_ptr=scales,
        zeros_ptr=zeros,
        scales_a_ptr=None,
        # Tensor sizes
        m_dim=metadata.m_dim,
        k_dim=metadata.k_dim,
        n_dim=metadata.n_dim,
        # Quantization paramers
        cxpr_w_nbits=metadata.weight_type.size_bits,
        cxpr_weight_bias=metadata.weight_bias,
        cxpr_group_size=metadata.group_size,
        cxpr_unpack_mask=metadata.unpack_mask,
        cxpr_elements_per_sample=metadata.elements_per_sample,
        # Strides
        matrix_a_stride_m=x.stride(0),
        matrix_a_stride_k=x.stride(1),
        matrix_b_stride_k=w_q_packed.stride(0),
        matrix_b_stride_n=w_q_packed.stride(1),
        matrix_c_stride_m=output.stride(0),
        matrix_c_stride_n=output.stride(1),
        meta_stride_g=scales.stride(0),
        meta_stride_n=scales.stride(1),
        # Data types
        cxpr_input_dtype=TORCH_DTYPE_TO_TRITON[metadata.input_dtype],
        cxpr_output_dtype=TORCH_DTYPE_TO_TRITON[metadata.output_dtype],
        cxpr_acc_dtype=TORCH_DTYPE_TO_TRITON[metadata.acc_dtype],
        cxpr_meta_dtype=TORCH_DTYPE_TO_TRITON[metadata.meta_dtype],
        # Metadata modes
        cxpr_channel_scale_mode=metadata.channel_scale_mode.value,
        cxpr_w_group_mode=metadata.weight_group_mode.value,
        cxpr_zero_is_scalar=metadata.zero_is_scalar,
        # Tuning parameters
        cxpr_data_contiguous=metadata.data_contiguous,
        cxpr_matrix_a_load_order=LoadOrder.MID.value,
        cxpr_matrix_a_eviction_policy=_get_matrix_a_eviction_policy(),
        cxpr_matrix_b_eviction_policy=_get_matrix_b_eviction_policy(),
        cxpr_meta_eviction_policy=_get_metadata_eviction_policy(),
        cxpr_swizzle_pid=True,
        cxpr_block_size_scale_a=tuning_parameters["cxpr_block_size_m"],
        cxpr_block_size_scale_b=tuning_parameters["cxpr_block_size_n"],
        **tuning_parameters,
    )


@dataclass
class ScaledMatmulMetadata:
    """Metadata for GEMM kernel."""

    m_dim: int
    k_dim: int
    n_dim: int
    data_contiguous: bool
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    acc_dtype: torch.dtype
    meta_dtype: torch.dtype
    channel_scale_mode: ChannelScaleMode
    weight_group_mode: WeightGroupMode


def scaled_gemm_launcher(
    output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    metadata: ScaledMatmulMetadata,
) -> None:
    """GEMM Triton Launcher."""
    grid = lambda parameters: (  # noqa: E731
        triton.cdiv(metadata.m_dim, parameters["cxpr_block_size_m"])
        * triton.cdiv(metadata.n_dim, parameters["cxpr_block_size_n"]),
    )

    zeros = torch.tensor([[]], dtype=torch.int32, device=a.device)

    tuning_parameters = _get_tuning_parameters()

    _gemm_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=output,
        scales_ptr=scale_b,
        zeros_ptr=zeros,
        scales_a_ptr=scale_a,
        # Tensor sizes
        m_dim=metadata.m_dim,
        k_dim=metadata.k_dim,
        n_dim=metadata.n_dim,
        # Quantization paramers
        cxpr_w_nbits=0,
        cxpr_weight_bias=0,
        cxpr_group_size=1,
        cxpr_unpack_mask=0,
        cxpr_elements_per_sample=1,
        cxpr_zero_is_scalar=True,
        # Strides
        matrix_a_stride_m=a.stride(0),
        matrix_a_stride_k=a.stride(1),
        matrix_b_stride_k=b.stride(0),
        matrix_b_stride_n=b.stride(1),
        matrix_c_stride_m=output.stride(0),
        matrix_c_stride_n=output.stride(1),
        meta_stride_g=1 if scale_b.numel() == 1 else scale_b.stride(0),
        meta_stride_n=1 if scale_b.numel() == 1 else scale_b.stride(1),
        # Data types
        cxpr_input_dtype=TORCH_DTYPE_TO_TRITON[metadata.input_dtype],
        cxpr_output_dtype=TORCH_DTYPE_TO_TRITON[metadata.output_dtype],
        cxpr_acc_dtype=TORCH_DTYPE_TO_TRITON[metadata.acc_dtype],
        cxpr_meta_dtype=TORCH_DTYPE_TO_TRITON[metadata.meta_dtype],
        # Metadata modes
        cxpr_channel_scale_mode=metadata.channel_scale_mode.value,
        cxpr_w_group_mode=metadata.weight_group_mode.value,
        # Tuning parameters
        cxpr_data_contiguous=metadata.data_contiguous,
        cxpr_matrix_a_load_order=LoadOrder.MID.value,
        cxpr_matrix_a_eviction_policy=_get_matrix_a_eviction_policy(),
        cxpr_matrix_b_eviction_policy=_get_matrix_b_eviction_policy(),
        cxpr_meta_eviction_policy=_get_metadata_eviction_policy(),
        cxpr_swizzle_pid=True,
        cxpr_block_size_scale_a=1 if scale_a.numel() == 1 else tuning_parameters["cxpr_block_size_m"],
        cxpr_block_size_scale_b=1 if scale_b.numel() == 1 else tuning_parameters["cxpr_block_size_n"],
        **tuning_parameters,
    )
