# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization implementation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _generate_dense_voxels_kernel(
    # Output tensors
    num_points_per_voxel_ptr: tl.tensor,
    point_features_ptr: tl.tensor,
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
    point_features_stride: int,
    points_stride: int,
    # Constexprs
    cxpr_block_size: tl.constexpr,
    cxpr_num_extra_features_padded: tl.constexpr,
) -> None:
    """Group valid points into dense voxels"""
    block_idx = tl.program_id(0)

    point_offsets = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    point_mask = point_offsets < num_points

    extra_offsets = 3 + tl.arange(0, cxpr_num_extra_features_padded)
    extra_mask = extra_offsets < 3 + num_extra_features

    point_x = tl.load(points_ptr + point_offsets * points_stride + 0, mask=point_mask, other=max_x + voxel_dim_x)
    point_y = tl.load(points_ptr + point_offsets * points_stride + 1, mask=point_mask, other=max_y + voxel_dim_y)
    point_z = tl.load(points_ptr + point_offsets * points_stride + 2, mask=point_mask, other=max_z + voxel_dim_z)
    # point_w = tl.load(points_ptr + point_offsets * points_stride + 3, mask=point_mask, other=0)
    point_extras = tl.load(points_ptr + point_offsets[:, None] * points_stride + extra_offsets[None, :], mask=point_mask[:, None] & extra_mask[None, :], other=0)

    # voxel_x = tl.math.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    # voxel_y = tl.math.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    # voxel_z = tl.math.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)
    voxel_x = tl.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)

    valid_x = (voxel_x >= 0) and (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) and (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) and (voxel_z < grid_dim_z)
    valid_point = ((point_mask and valid_x) and valid_y) and valid_z

    # Triton atomics do not take masks... manually protect
    # flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x
    flat_voxel_idx = (voxel_z * grid_dim_x * grid_dim_y) + (voxel_y * grid_dim_x) + voxel_x
    # flat_voxel_idx = tl.minimum(flat_voxel_idx, tl.full((cxpr_block_size,), max_num_voxels - 1, tl.int32))
    valid_voxel = flat_voxel_idx >= 0 and flat_voxel_idx < max_num_voxels
    flat_voxel_idx = tl.minimum(tl.maximum(flat_voxel_idx, 0), max_num_voxels - 1)
    # flat_voxel_idx = tl.clamp(flat_voxel_idx, min=0, max=max_num_voxels - 1)

    # point_idx_in_voxel = tl.atomic_add(num_points_per_voxel_ptr + flat_voxel_idx, valid_point.to(tl.int32))
    point_idx_in_voxel = tl.atomic_add(num_points_per_voxel_ptr + flat_voxel_idx, 1, mask=valid_point & valid_voxel)

    # output_idx = flat_voxel_idx * max_num_points_per_voxel + point_idx_in_voxel
    output_idx = flat_voxel_idx * point_features_stride + point_idx_in_voxel
    output_mask = valid_point & point_idx_in_voxel < max_num_points_per_voxel
    # tl.store(point_features_ptr + output_idx * 4 + 0, point_x, mask=output_mask)
    # tl.store(point_features_ptr + output_idx * 4 + 1, point_y, mask=output_mask)
    # tl.store(point_features_ptr + output_idx * 4 + 2, point_z, mask=output_mask)
    # tl.store(point_features_ptr + output_idx * 4 + 3, point_w, mask=output_mask)
    tl.store(point_features_ptr + output_idx + 0, point_x, mask=output_mask)
    tl.store(point_features_ptr + output_idx + 1, point_y, mask=output_mask)
    tl.store(point_features_ptr + output_idx + 2, point_z, mask=output_mask)
    tl.store(point_features_ptr + output_idx[:, None] + extra_offsets[None, :], point_extras, mask=output_mask[:, None] & extra_mask[None, :])


def voxelization_launcher(
    num_points_per_voxel: torch.Tensor,
    point_features: torch.Tensor,
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> None:
    num_points, num_features = points.shape

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

    def grid(meta):
        return (triton.cdiv(num_points, meta["cxpr_block_size"]),)

    _generate_dense_voxels_kernel[grid](
        # Outputs
        num_points_per_voxel_ptr=num_points_per_voxel,
        point_features_ptr=point_features,
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
        point_features_stride=point_features.stride(0),
        points_stride=points.stride(0),
        cxpr_block_size=32,
        cxpr_num_extra_features_padded=triton.next_power_of_2(num_features),
    )
