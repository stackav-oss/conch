# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Quick cumsum kernel for 3D voxel grids."""

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
        image_feats_stride: The stride of the image features tensor.
        geom_feats_stride: The stride of the geometry features tensor.
        output_batch_stride: The stride of the output tensor for the batch dimension.
        output_z_stride: The stride of the output tensor for the z dimension.
        output_x_stride: The stride of the output tensor for the x dimension.
        output_y_stride: The stride of the output tensor for the y dimension.
        cxpr_num_channels_padded: The number of channels padded to the next power of 2.
        cxpr_block_size: The block size for processing points in parallel.
    """
    # What is the index of the interval this program is processing?
    interval_index = tl.program_id(0)

    # Load current interval start and length
    interval_start = tl.load(interval_starts_ptr + interval_index)
    interval_length = tl.load(interval_lengths_ptr + interval_index)

    # Offsets and masks for the channels
    channel_offsets = tl.arange(0, cxpr_num_channels_padded)
    channel_mask = channel_offsets < num_channels

    # Accumulator for the sum of image features for each channel for all points in the interval
    output = tl.zeros([cxpr_num_channels_padded], dtype=image_feats_ptr.dtype.element_ty)

    # Calculate the pointer to the start of the image features for the current interval
    current_image_feats_ptr = image_feats_ptr + interval_start * image_feats_stride + channel_offsets[None, :]

    # Iterate blockwise over the points in the interval
    for block_index in range(tl.cdiv(interval_length, cxpr_block_size)):
        # Offsets for the current block
        block_offsets = block_index * cxpr_block_size + tl.arange(0, cxpr_block_size)
        # Mask out any indices that are out of bounds for the interval length
        block_mask = block_offsets < interval_length

        # Load image features, shape: (cxpr_block_size, num_channels)
        image_feats = tl.load(
            current_image_feats_ptr + block_offsets[:, None] * image_feats_stride,
            mask=block_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )

        # Calculate sum of image features for the current block, per channel
        # Shape: (cxpr_num_channels_padded,)
        output += tl.sum(image_feats, axis=0)

    # Load geometry coordinates for the first point in this interval
    # Note: all points in the interval share the same geom_feats, so we only need to load the first point's geom_feats.
    current_geom_feats_ptr = geom_feats_ptr + interval_start * geom_feats_stride
    geom_x = tl.load(current_geom_feats_ptr + 0)  # X coordinate
    geom_y = tl.load(current_geom_feats_ptr + 1)  # Y coordinate
    geom_z = tl.load(current_geom_feats_ptr + 2)  # Z coordinate
    geom_b = tl.load(current_geom_feats_ptr + 3)  # Batch index

    # Calculate output tensor offset for shape [batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels]
    batch_offset = geom_b * output_batch_stride
    z_offset = geom_z * output_z_stride
    x_offset = geom_x * output_x_stride
    y_offset = geom_y * output_y_stride

    # Store the accumulated output for the current interval
    tl.store(
        output_ptr + batch_offset + z_offset + x_offset + y_offset + channel_offsets,
        output,
        mask=channel_mask,
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

    # Process each interval in parallel
    grid = (num_intervals,)

    _bev_pool_kernel[grid](
        # Pointers to tensors
        output_ptr=output,
        image_feats_ptr=image_feats,
        geom_feats_ptr=geom_feats,
        interval_starts_ptr=interval_starts,
        interval_lengths_ptr=interval_lengths,
        # Scalars
        num_channels=num_channels,
        # Strides
        image_feats_stride=image_feats.stride(0),
        geom_feats_stride=geom_feats.stride(0),
        output_batch_stride=output.stride(0),
        output_z_stride=output.stride(1),
        output_x_stride=output.stride(2),
        output_y_stride=output.stride(3),
        # Constexprs
        cxpr_num_channels_padded=triton.next_power_of_2(num_channels),
        # TODO(jmanning): Autotune?
        # The tricky thing here is the optimal block size depends on the average/maximum interval length,
        # not the number of intervals or number of points.
        # It also just may depend on the platform/device what the optimal block size is.
        cxpr_block_size=64,
    )
