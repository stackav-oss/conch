# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization implementation."""

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _generate_dense_voxels_kernel(
    # Output tensors
    dense_num_points_per_voxel_ptr: tl.tensor,
    dense_point_features_ptr: tl.tensor,
    # Input tensors
    points_ptr: tl.tensor,
    # Scalars
    num_points: int,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    voxel_dim_x: float,
    voxel_dim_y: float,
    voxel_dim_z: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    max_num_points_per_voxel: int,
    max_num_voxels: int,
    num_extra_features: int,
    # Strides
    point_features_voxel_stride: int,
    point_features_point_stride: int,
    points_stride: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
    cxpr_num_extra_features_padded: tl.constexpr,
) -> None:
    """Group points into dense voxels."""
    # What block of the data is this program processing?
    block_index = tl.program_id(0)

    # Process a block of points at a time
    block_offsets = block_index * cxpr_block_size + tl.arange(0, cxpr_block_size)
    block_mask = block_offsets < num_points

    # First three features are always XYZ
    extra_features_offsets = 3 + tl.arange(0, cxpr_num_extra_features_padded)
    extra_features_mask = extra_features_offsets < 3 + num_extra_features

    # Load x, y, z features for the current block
    # Shape: (cxpr_block_size,)
    block_x = tl.load(points_ptr + block_offsets * points_stride + 0, mask=block_mask, other=max_x + voxel_dim_x)
    block_y = tl.load(points_ptr + block_offsets * points_stride + 1, mask=block_mask, other=max_y + voxel_dim_y)
    block_z = tl.load(points_ptr + block_offsets * points_stride + 2, mask=block_mask, other=max_z + voxel_dim_z)
    # Load extra features for the current block
    # Shape: (cxpr_block_size, cxpr_num_extra_features_padded)
    block_extras = tl.load(
        points_ptr + block_offsets[:, None] * points_stride + extra_features_offsets[None, :],
        mask=block_mask[:, None] & extra_features_mask[None, :],
        other=0,
    )

    # print("block_x = ", block_x)
    # print("block_y = ", block_y)
    # print("block_z = ", block_z)

    # Calculate voxel indices for each point
    # Shape: (cxpr_block_size,)
    voxel_x = tl.floor((block_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((block_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((block_z - min_z) / voxel_dim_z).to(tl.int32)

    # print("voxel_x = ", voxel_x)
    # print("voxel_y = ", voxel_y)
    # print("voxel_z = ", voxel_z)

    # Determine if voxel indices are valid
    valid_x = (voxel_x >= 0) and (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) and (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) and (voxel_z < grid_dim_z)
    valid_coordinate_mask = ((block_mask and valid_x) and valid_y) and valid_z

    # print("valid_x = ", valid_x)
    # print("valid_y = ", valid_y)
    # print("valid_z = ", valid_z)

    # print("grid_dim_x = ", grid_dim_x)
    # print("grid_dim_y = ", grid_dim_y)
    # print("grid_dim_z = ", grid_dim_z)

    # "Flatten" voxel indices from 3D -> 1D
    # Shape: (cxpr_block_size,)
    flat_voxel_indices = (voxel_z * grid_dim_x * grid_dim_y) + (voxel_y * grid_dim_x) + voxel_x
    # print("flat_voxel_indices = BEFORE ", flat_voxel_indices)
    # Mask out any invalid indices
    valid_voxel_mask = flat_voxel_indices >= 0 and flat_voxel_indices < max_num_voxels
    # Clamp offsets between (0, max_num_voxels - 1) so that we don't accidentally read invalid addresses
    # flat_voxel_indices = tl.minimum(tl.maximum(flat_voxel_indices, 0), max_num_voxels - 1)
    # print("flat_voxel_indices AFTER = ", flat_voxel_indices)

    # print("valid_coordinate_mask = ", valid_coordinate_mask)
    # print("valid_voxel_mask = ", valid_voxel_mask)
    # print("atomic_mask = ", valid_coordinate_mask & valid_voxel_mask)

    # For each flat voxel index that corresponds to a point in the current block, perform an atomic increment
    # of the number of points in that voxel. This operation also returns the previous number of points in each
    # voxel, which we will use for storing the point features (x, y, z, ...) in the output
    # Shape: (cxpr_block_size,)
    indices_in_voxel = tl.atomic_add(dense_num_points_per_voxel_ptr + flat_voxel_indices, 1, mask=valid_coordinate_mask & valid_voxel_mask)

    # print("indices_in_voxel = ", indices_in_voxel)

    # Use the previous number of points in each voxel and the flat voxel indices to find the appropriate offsets
    # to write the point features to the output tensor
    output_offsets = flat_voxel_indices * point_features_voxel_stride + indices_in_voxel * point_features_point_stride
    points_per_voxel_mask = indices_in_voxel < max_num_points_per_voxel
    output_mask = valid_coordinate_mask & points_per_voxel_mask

    # print("output_offsets = ", output_offsets)
    # print("output_mask = ", output_mask)
    # print("extra_output_offsets = ", output_offsets[:, None] + extra_features_offsets[None, :])

    # Store x, y, z features for the current block
    tl.store(dense_point_features_ptr + output_offsets + 0, block_x, mask=output_mask)
    tl.store(dense_point_features_ptr + output_offsets + 1, block_y, mask=output_mask)
    tl.store(dense_point_features_ptr + output_offsets + 2, block_z, mask=output_mask)
    # Store extra features for the current block
    tl.store(
        dense_point_features_ptr + output_offsets[:, None] + extra_features_offsets[None, :],
        block_extras,
        mask=output_mask[:, None] & extra_features_mask[None, :],
    )


@triton.jit
def _generate_sparse_voxels_kernel(
    # Output tensors
    num_filled_voxels_ptr: tl.tensor,
    num_points_per_voxel_ptr: tl.tensor,
    point_features_ptr: tl.tensor,
    voxel_indices_ptr: tl.tensor,
    # Input tensors
    dense_point_features_ptr: tl.tensor,
    dense_num_points_per_voxel_ptr: tl.tensor,
    # Scalars
    grid_dim_x: int,
    grid_dim_y: int,
    max_num_points_per_voxel: int,
    max_num_voxels: int,
    # Strides
    voxel_indices_stride: int,
    dense_point_features_stride: int,
    point_features_stride: int,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Convert dense voxels into sparse/contiguous non-empty voxels.
    output voxel/points ordering is nondeterministic.
    """
    block_index = tl.program_id(0)

    flat_voxel_indices = block_index * cxpr_block_size + tl.arange(0, cxpr_block_size)
    flat_voxel_mask = flat_voxel_indices < max_num_voxels

    num_points_in_voxel = tl.load(
        dense_num_points_per_voxel_ptr + flat_voxel_indices, mask=flat_voxel_mask, other=0
    )
    num_points_in_voxel = tl.minimum(num_points_in_voxel, max_num_points_per_voxel)
    valid_voxel_mask = num_points_in_voxel > 0

    current_voxel_offsets = tl.atomic_add(num_filled_voxels_ptr + tl.zeros_like(valid_voxel_mask), 1, mask=valid_voxel_mask)

    # store num_points_per_voxel with clipping
    tl.store(num_points_per_voxel_ptr + current_voxel_offsets, num_points_in_voxel, mask=valid_voxel_mask)

    # convert flat voxel index to 3d coordinates
    voxel_x = flat_voxel_indices % grid_dim_x
    voxel_y = (flat_voxel_indices // grid_dim_x) % grid_dim_y
    voxel_z = flat_voxel_indices // (grid_dim_y * grid_dim_x)

    # store 3d indices
    tl.store(voxel_indices_ptr + current_voxel_offsets * voxel_indices_stride + 0, voxel_x, mask=valid_voxel_mask)
    tl.store(voxel_indices_ptr + current_voxel_offsets * voxel_indices_stride + 1, voxel_y, mask=valid_voxel_mask)
    tl.store(voxel_indices_ptr + current_voxel_offsets * voxel_indices_stride + 2, voxel_z, mask=valid_voxel_mask)

    # copy from nvidia opensource code, index is padded to int4 type
    # tl.store(voxel_indices_ptr + current_voxel_offsets * 4 + 3, 0, mask=valid_voxel_mask)

    # store all feature points, even if they are 0 because Triton
    for point_idx in range(max_num_points_per_voxel):
        input_idx = flat_voxel_indices * max_num_points_per_voxel + point_idx
        point_x = tl.load(dense_point_features_ptr + input_idx * dense_point_features_stride + 0, mask=valid_voxel_mask)
        point_y = tl.load(dense_point_features_ptr + input_idx * dense_point_features_stride + 1, mask=valid_voxel_mask)
        point_z = tl.load(dense_point_features_ptr + input_idx * dense_point_features_stride + 2, mask=valid_voxel_mask)
        # point_w = tl.load(dense_point_features_ptr + input_idx * dense_point_features_stride + 3, mask=valid_voxel_mask)

        output_idx = current_voxel_offsets * max_num_points_per_voxel + point_idx
        tl.store(point_features_ptr + output_idx * point_features_stride + 0, point_x, mask=valid_voxel_mask)
        tl.store(point_features_ptr + output_idx * point_features_stride + 1, point_y, mask=valid_voxel_mask)
        tl.store(point_features_ptr + output_idx * point_features_stride + 2, point_z, mask=valid_voxel_mask)
        # tl.store(point_features_ptr + output_idx * 4 + 3, point_w, mask=valid_voxel_mask)


def dense_voxelization_launcher(
    dense_num_points_per_voxel: torch.Tensor,
    dense_point_features: torch.Tensor,
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> None:
    num_points, num_features = points.shape

    # print(f"{num_points_per_voxel = }")

    assert num_features > 3
    num_features -= 3

    (
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
    ) = coordinate_range

    voxel_size_x, voxel_size_y, voxel_size_z = voxel_size

    grid_size_x = int((max_x - min_x) / voxel_size_x)
    grid_size_y = int((max_y - min_y) / voxel_size_y)
    grid_size_z = int((max_z - min_z) / voxel_size_z)

    # print(f"{grid_size_x = }")
    # print(f"{grid_size_y = }")
    # print(f"{grid_size_z = }")

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (triton.cdiv(num_points, meta["cxpr_block_size"]),)

    # print(f"{point_features.shape = }")
    # print(f"{point_features.stride(0) = }")
    # print(f"{point_features.stride(1) = }")

    _generate_dense_voxels_kernel[grid](
        # Outputs
        dense_num_points_per_voxel_ptr=dense_num_points_per_voxel,
        dense_point_features_ptr=dense_point_features,
        # Inputs
        points_ptr=points,
        # Scalars
        num_points=num_points,
        min_x=min_x,
        min_y=min_y,
        min_z=min_z,
        max_x=max_x,
        max_y=max_y,
        max_z=max_z,
        voxel_dim_x=voxel_size_x,
        voxel_dim_y=voxel_size_y,
        voxel_dim_z=voxel_size_z,
        grid_dim_x=grid_size_x,
        grid_dim_y=grid_size_y,
        grid_dim_z=grid_size_z,
        max_num_points_per_voxel=max_points_per_voxel,
        max_num_voxels=max_voxels,
        num_extra_features=num_features,
        # Strides
        point_features_voxel_stride=dense_point_features.stride(0),
        point_features_point_stride=dense_point_features.stride(1),
        points_stride=points.stride(0),
        # Constexprs
        cxpr_block_size=32,
        cxpr_num_extra_features_padded=triton.next_power_of_2(num_features),
    )

    # torch.cuda.synchronize()

    # print(f"{num_points_per_voxel = }")


def sparse_voxelization_launcher(
    num_filled_voxels: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    point_features: torch.Tensor,
    voxel_indices: torch.Tensor,
    dense_num_points_per_voxel: torch.Tensor,
    dense_point_features: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
) -> None:
    max_voxels, max_points_per_voxel, num_features = dense_point_features.shape
    # num_points, _ = point_features

    # print(f"{num_points_per_voxel = }")

    assert num_features > 3
    num_features -= 3

    (
        min_x,
        min_y,
        min_z,
        max_x,
        max_y,
        max_z,
    ) = coordinate_range

    voxel_size_x, voxel_size_y, voxel_size_z = voxel_size

    grid_size_x = int((max_x - min_x) / voxel_size_x)
    grid_size_y = int((max_y - min_y) / voxel_size_y)
    grid_size_z = int((max_z - min_z) / voxel_size_z)

    # print(f"{grid_size_x = }")
    # print(f"{grid_size_y = }")
    # print(f"{grid_size_z = }")

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (triton.cdiv(max_voxels, meta["cxpr_block_size"]),)

    # print(f"{point_features.shape = }")
    # print(f"{point_features.stride(0) = }")
    # print(f"{point_features.stride(1) = }")

    _generate_sparse_voxels_kernel[grid](
        # Output tensors
        num_filled_voxels_ptr=num_filled_voxels,
        num_points_per_voxel_ptr=num_points_per_voxel,
        point_features_ptr=point_features,
        voxel_indices_ptr=voxel_indices,
        # Input tensors
        dense_num_points_per_voxel_ptr=dense_num_points_per_voxel,
        dense_point_features_ptr=dense_point_features,
        # Scalars
        # num_points=num_points,
        # min_x=min_x,
        # min_y=min_y,
        # min_z=min_z,
        # max_x=max_x,
        # max_y=max_y,
        # max_z=max_z,
        # voxel_dim_x=voxel_size_x,
        # voxel_dim_y=voxel_size_y,
        # voxel_dim_z=voxel_size_z,
        grid_dim_x=grid_size_x,
        grid_dim_y=grid_size_y,
        # grid_dim_z=grid_size_z,
        max_num_points_per_voxel=max_points_per_voxel,
        max_num_voxels=max_voxels,
        # num_extra_features=num_features,
        # Strides
        voxel_indices_stride=voxel_indices.stride(0),
        dense_point_features_stride=dense_point_features.stride(0),
        point_features_stride=point_features.stride(0),
        # point_features_voxel_stride=point_features.stride(0),
        # point_features_point_stride=point_features.stride(1),
        # points_stride=points.stride(0),
        # Constexprs
        cxpr_block_size=32,
        # cxpr_num_extra_features_padded=triton.next_power_of_2(num_features),
    )

    # torch.cuda.synchronize()

    # print(f"{num_points_per_voxel = }")
