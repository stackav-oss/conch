# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Quick cumsum kernel for 3D voxel grids."""

import torch
import triton
import triton.language as tl


# @triton.autotune(  # type: ignore[misc]
#     configs=[
#         triton.Config({"cxpr_block_size": 16}),
#         triton.Config({"cxpr_block_size": 32}),
#         triton.Config({"cxpr_block_size": 64}),
#         triton.Config({"cxpr_block_size": 128}),
#         triton.Config({"cxpr_block_size": 256}),
#     ],
#     key=["num_intervals"],
# )
@triton.jit  # type: ignore[misc]
def _quick_cumsum_kernel(
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

    For each image feature that belongs in a pooled point, we sum the features
    and store the result in the output tensor at the corresponding grid cell.

    image_feats: The input image features, shape: (num_points, num_channels).
    geom_feats: The input coordinates, shape: (num_points, 4).

    The geom_feats represent the voxel coordinates in the grid:
    - geom_feats[:, 0]: batch index
    - geom_feats[:, 1]: z coordinate
    - geom_feats[:, 2]: x coordinate
    - geom_feats[:, 3]: y coordinate

    interval_starts: Starting position for pooled point, shape: (num_intervals,).
    interval_lengths: How many points in each pooled point, shape: (num_intervals,

    So for each interval, all pooled points will have the same geom_feats,
    which means they will be pooled into the same grid cell.

    We read the image features for each interval and sum them up,
    then store the result in the output tensor at the corresponding grid cell.
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
    geom_b = tl.load(current_geom_feats_ptr + 0)  # batch index
    geom_z = tl.load(current_geom_feats_ptr + 1)  # z coordinate
    geom_x = tl.load(current_geom_feats_ptr + 2)  # x coordinate
    geom_y = tl.load(current_geom_feats_ptr + 3)  # y coordinate

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


def quick_cumsum_launcher(
    image_feats: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
) -> torch.Tensor:
    """Launch the quick cumulative sum kernel.

    Args:
        image_feats: input image features, shape: (num_points, num_channels).
        geom_feats: input coordinates, shape: (num_points, 4).
        interval_starts: starting position for pooled point, shape: (num_intervals,).
        interval_lengths: how many points in each pooled point, shape: (num_intervals,).
    """
    _, num_channels = image_feats.shape
    num_intervals = interval_lengths.size(0)

    # Create output tensor
    output = torch.zeros(
        (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels),
        dtype=image_feats.dtype,
        device=image_feats.device,
    )

    # Process each interval in parallel
    grid = (num_intervals,)

    _quick_cumsum_kernel[grid](
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

    return output
