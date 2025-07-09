# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Quick cumsum kernel for 3D voxel grids."""

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit  # type: ignore[misc]
def _bev_pool_kernel(
    # Pointers to tensors
    output_ptr: tl.tensor,
    image_feats_ptr: tl.tensor,
    geom_feats_ptr: tl.tensor,
    interval_starts_ptr: tl.tensor,
    interval_lengths_ptr: tl.tensor,
    # Scalars
    num_channels: tl.int32,
    num_intervals: tl.int32,
    # Strides
    image_feats_stride: tl.int32,
    geom_feats_stride: tl.int32,
    output_batch_stride: tl.int32,
    output_z_stride: tl.int32,
    output_x_stride: tl.int32,
    output_y_stride: tl.int32,
    # Constexprs
    cxpr_num_channels_padded: tl.constexpr,
    cxpr_block_size: tl.constexpr,
) -> None:
    """Kernel for quick cumulative sum.

    The geom_feats represent the voxel coordinates in the grid:
    - geom_feats[:, 0]: X coordinate
    - geom_feats[:, 1]: Y coordinate
    - geom_feats[:, 2]: Z coordinate
    - geom_feats[:, 3]: Batch index

    So for each interval, all pooled points will have the same geom_feats,
    which means they will be pooled into the same grid cell. We read the image features
    for each interval and sum them up, then store the result in the output tensor at
    the corresponding grid cell.

    Args:
        output_ptr: Pointer to the output tensor, shape: (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels).
        image_feats_ptr: Pointer to the input image features, shape: (num_points, num_channels).
        geom_feats_ptr: Pointer to the input coordinates, shape: (num_points, 4).
        interval_starts_ptr: Pointer to the starting positions for pooled points, shape: (num_intervals,).
        interval_lengths_ptr: Pointer to the lengths of each pooled point, shape: (num_intervals,).
        num_channels: The number of channels in the image features.
        num_intervals: The number of intervals to process.
        image_feats_stride: The stride of the image features tensor.
        geom_feats_stride: The stride of the geometry features tensor.
        output_batch_stride: The stride of the output tensor for the batch dimension.
        output_z_stride: The stride of the output tensor for the z dimension.
        output_x_stride: The stride of the output tensor for the x dimension.
        output_y_stride: The stride of the output tensor for the y dimension.
        cxpr_num_channels_padded: The number of channels padded to the next power of 2.
        cxpr_block_size: The block size for processing intervals in parallel.
    """
    # What is the starting index of the block of intervals this program is processing?
    interval_block_start = tl.program_id(0) * cxpr_block_size
    # Offsets for each interval in the block
    interval_block_offsets = interval_block_start + tl.arange(0, cxpr_block_size)
    # Mask out-of-bounds intervals
    interval_block_mask = interval_block_offsets < num_intervals

    # Load start and length for each interval in the block
    interval_starts = tl.load(interval_starts_ptr + interval_block_offsets, mask=interval_block_mask, other=0)
    interval_lengths = tl.load(interval_lengths_ptr + interval_block_offsets, mask=interval_block_mask, other=0)

    # Offsets and masks for the channels
    channel_offsets = tl.arange(0, cxpr_num_channels_padded)
    channel_mask = channel_offsets < num_channels

    # Combine interval block mask with channel mask
    output_mask = interval_block_mask[:, None] & channel_mask[None, :]

    # Accumulator for the sum of image features for each channel for all intervals in the block
    output = tl.zeros([cxpr_block_size, cxpr_num_channels_padded], dtype=image_feats_ptr.dtype.element_ty)

    # Calculate the pointer to the start of the image features for the current block of intervals
    # Shape: (cxpr_block_size, cxpr_num_channels_padded)
    current_image_feats_ptr = image_feats_ptr + interval_starts[:, None] * image_feats_stride + channel_offsets[None, :]

    # Determine the maximum interval length in the block
    # This is used to determine how many points we need to process in the block
    max_interval_length = tl.max(interval_lengths, axis=0)

    # Iterate over the max number of points in any interval in the block
    for point_index in range(max_interval_length):
        # Mask for intervals where this point index is valid
        index_mask = point_index < interval_lengths

        # Load image features for the current point, shape: (cxpr_block_size, cxpr_num_channels_padded)
        image_feats = tl.load(
            current_image_feats_ptr + point_index * image_feats_stride,
            mask=index_mask[:, None] & output_mask,
            other=0.0,
        )

        # Accumulate the image features into the output tensor
        output += image_feats

    # Load geometry coordinates for the first point in each interval
    # Note: all points in each interval share the same geom_feats, so we only need to load the first point's geom_feats.
    # geom_{x|y|z|b} shape: (cxpr_block_size,)
    current_geom_feats_ptrs = geom_feats_ptr + interval_starts[:, None] * geom_feats_stride
    geom_x = tl.load(current_geom_feats_ptrs + 0)  # X coordinates
    geom_y = tl.load(current_geom_feats_ptrs + 1)  # Y coordinates
    geom_z = tl.load(current_geom_feats_ptrs + 2)  # Z coordinates
    geom_b = tl.load(current_geom_feats_ptrs + 3)  # Batch indices

    # Calculate output tensor offsets for shape [batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels]
    batch_offsets = geom_b * output_batch_stride
    z_offsets = geom_z * output_z_stride
    x_offsets = geom_x * output_x_stride
    y_offsets = geom_y * output_y_stride

    # Store the accumulated output for the current interval
    tl.store(
        output_ptr + batch_offsets + z_offsets + x_offsets + y_offsets + channel_offsets[None, :],
        output,
        mask=output_mask,
    )


@triton.jit  # type: ignore[misc]
def _bev_pool_backward_kernel(
    # Pointers to tensors
    x_grad_ptr: tl.tensor,
    grad_output_ptr: tl.tensor,
    geom_feats_ptr: tl.tensor,
    interval_starts_ptr: tl.tensor,
    interval_lengths_ptr: tl.tensor,
    # Scalars
    num_channels: tl.int32,
    num_intervals: tl.int32,
    # Strides
    x_grad_stride: tl.int32,
    grad_output_batch_stride: tl.int32,
    grad_output_z_stride: tl.int32,
    grad_output_x_stride: tl.int32,
    grad_output_y_stride: tl.int32,
    geom_feats_stride: tl.int32,
    # Constexprs
    cxpr_num_channels_padded: tl.constexpr,
    cxpr_block_size: tl.constexpr,
) -> None:
    """Kernel for backward pass of quick cumulative sum.

    This kernel computes the gradient with respect to the input image features
    by accumulating gradients from the output tensor.

    Args:
        x_grad_ptr: Pointer to the gradient with respect to the input image features.
        grad_output_ptr: Pointer to the gradient of the output tensor.
        geom_feats_ptr: Pointer to the geometry features tensor.
        interval_starts_ptr: Pointer to the starting positions for pooled points.
        interval_lengths_ptr: Pointer to the lengths of each pooled point.
        num_channels: The number of channels in the image features.
        num_intervals: The number of intervals to process.
        x_grad_stride: The stride of the x_grad tensor.
        grad_output_batch_stride: The stride of the grad_output tensor for the batch dimension.
        grad_output_z_stride: The stride of the grad_output tensor for the z dimension.
        grad_output_x_stride: The stride of the grad_output tensor for the x dimension.
        grad_output_y_stride: The stride of the grad_output tensor for the y dimension.
        geom_feats_stride: The stride of the geometry features tensor.
        cxpr_num_channels_padded: The number of channels padded to the next power of 2.
        cxpr_block_size: The block size for processing points in parallel.
    """
    # What is the starting index of the block of intervals this program is processing?
    interval_block_start = tl.program_id(0) * cxpr_block_size
    # Offsets for each interval in the block
    interval_block_offsets = interval_block_start + tl.arange(0, cxpr_block_size)
    # Mask out-of-bounds intervals
    interval_block_mask = interval_block_offsets < num_intervals

    # Load start and length for each interval in the block
    interval_starts = tl.load(interval_starts_ptr + interval_block_offsets, mask=interval_block_mask, other=0)
    interval_lengths = tl.load(interval_lengths_ptr + interval_block_offsets, mask=interval_block_mask, other=0)

    # Offsets and masks for the channels
    channel_offsets = tl.arange(0, cxpr_num_channels_padded)
    channel_mask = channel_offsets < num_channels

    # Combine interval block mask with channel mask
    output_mask = interval_block_mask[:, None] & channel_mask[None, :]

    # Load geometry coordinates for the first point in each interval
    # Note: all points in each interval share the same geom_feats, so we only need to load the first point's geom_feats.
    current_geom_feats_ptrs = geom_feats_ptr + interval_starts[:, None] * geom_feats_stride
    # geom_{x|y|z|b} shape: (cxpr_block_size,)
    geom_x = tl.load(current_geom_feats_ptrs + 0)  # X coordinates
    geom_y = tl.load(current_geom_feats_ptrs + 1)  # Y coordinates
    geom_z = tl.load(current_geom_feats_ptrs + 2)  # Z coordinates
    geom_b = tl.load(current_geom_feats_ptrs + 3)  # Batch indices

    # Offsets for the entry in the grad_output tensor for each interval
    grad_output_offsets = (
        geom_b * grad_output_batch_stride
        + geom_z * grad_output_z_stride
        + geom_x * grad_output_x_stride
        + geom_y * grad_output_y_stride
    )

    # Load gradient output, shape: (cxpr_block_size, cxpr_num_channels_padded)
    grad_output = tl.load(
        grad_output_ptr + grad_output_offsets + channel_offsets[None, :],
        mask=output_mask,
        other=0.0,
    )

    # Pointer to the start of the output for this block
    current_x_grad_ptr = x_grad_ptr + interval_starts[:, None] * x_grad_stride + channel_offsets[None, :]

    # Determine the maximum interval length in the block
    # This is used to determine how many points we need to process in the block
    max_interval_length = tl.max(interval_lengths, axis=0)

    # Iterate over the max number of points in any interval in the block
    for point_index in range(max_interval_length):
        # Mask for intervals where this point index is valid
        index_mask = point_index < interval_lengths

        # Store gradients for the current point, shape: (cxpr_block_size, cxpr_num_channels_padded)
        tl.store(
            current_x_grad_ptr + point_index * x_grad_stride,
            grad_output,
            mask=index_mask[:, None] & output_mask,
        )


def bev_pool_launcher(
    output: torch.Tensor,
    image_feats: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> None:
    """Launch the quick cumulative sum kernel.

    Args:
        output: output tensor, shape: (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels).
        image_feats: input image features, shape: (num_points, num_channels).
        geom_feats: input coordinates in form (B, X, Y, Z), shape: (num_points, 4).
        interval_starts: starting position for pooled point, shape: (num_intervals,).
        interval_lengths: how many points in each pooled point, shape: (num_intervals,).
    """
    _, num_channels = image_feats.shape
    num_intervals = interval_lengths.size(0)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        # Process each interval in parallel, blockwise
        return (triton.cdiv(num_intervals, meta["cxpr_block_size"]),)

    _bev_pool_kernel[grid](
        # Pointers to tensors
        output_ptr=output,
        image_feats_ptr=image_feats,
        geom_feats_ptr=geom_feats,
        interval_starts_ptr=interval_starts,
        interval_lengths_ptr=interval_lengths,
        # Scalars
        num_channels=num_channels,
        num_intervals=num_intervals,
        # Strides
        image_feats_stride=image_feats.stride(0),
        geom_feats_stride=geom_feats.stride(0),
        output_batch_stride=output.stride(0),
        output_z_stride=output.stride(1),
        output_x_stride=output.stride(2),
        output_y_stride=output.stride(3),
        # Constexprs
        cxpr_num_channels_padded=triton.next_power_of_2(num_channels),
        # TODO(jmanning): We _could_ autotune based on the number of intervals,
        # but that would likely trigger many recompilations.
        cxpr_block_size=64,
    )


def bev_pool_backward_launcher(
    x_grad: torch.Tensor,
    grad_output: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> None:
    """Launch the backward pass for the BEV Pool operation.

    Args:
        x_grad: gradient with respect to the input image features, shape: (num_points, num_channels).
        grad_output: gradient of the output, shape: (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels).
        geom_feats: input coordinates in form (B, X, Y, Z), shape: (num_points, 4).
        interval_starts: starting position for pooled point, shape: (num_intervals,).
        interval_lengths: how many points in each pooled point, shape: (num_intervals,).
    """
    num_intervals = interval_starts.size(0)
    _, num_channels = x_grad.shape

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        # Process each interval in parallel, blockwise
        return (triton.cdiv(num_intervals, meta["cxpr_block_size"]),)

    _bev_pool_backward_kernel[grid](
        # Pointers to tensors
        x_grad_ptr=x_grad,
        grad_output_ptr=grad_output,
        geom_feats_ptr=geom_feats,
        interval_starts_ptr=interval_starts,
        interval_lengths_ptr=interval_lengths,
        # Scalars
        num_channels=num_channels,
        num_intervals=num_intervals,
        # Strides
        x_grad_stride=x_grad.stride(0),
        grad_output_batch_stride=grad_output.stride(0),
        grad_output_z_stride=grad_output.stride(1),
        grad_output_x_stride=grad_output.stride(2),
        grad_output_y_stride=grad_output.stride(3),
        geom_feats_stride=geom_feats.stride(0),
        # Constexprs
        cxpr_num_channels_padded=triton.next_power_of_2(num_channels),
        cxpr_block_size=64,
    )
