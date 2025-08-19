# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton point cloud voxeliation kernels."""

import torch
import triton
import triton.language as tl

@triton.jit
def filter_and_label_points_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_points: int,
    num_features_per_point: int,
    # parameters
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
    max_num_voxels: int,
    # output
    point_voxel_indices_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Filter valid points and label each with a voxel index.

    Args:
        points_ptr: input points, shape (num_points, num_features_per_point).
        voxelization parameters
        point_voxel_indices_ptr: output per point flattened voxel indices, shape (num_points).
    """
    block_idx = tl.program_id(axis=0)
    point_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    point_mask = point_idx < num_points

    point_x = tl.load(points_ptr + point_idx * num_features_per_point + 0, mask=point_mask, other=max_x + voxel_dim_x)
    point_y = tl.load(points_ptr + point_idx * num_features_per_point + 1, mask=point_mask, other=max_y + voxel_dim_y)
    point_z = tl.load(points_ptr + point_idx * num_features_per_point + 2, mask=point_mask, other=max_z + voxel_dim_z)

    voxel_x = tl.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)

    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = point_mask & valid_x & valid_y & valid_z

    flat_voxel_idx = tl.where(valid_point, ((voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x), max_num_voxels)
    tl.store(point_voxel_indices_ptr + point_idx, flat_voxel_idx, mask=point_mask)


@triton.jit
def generate_dense_voxels_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_points: int,
    # parameters
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
    # output
    dense_num_points_per_voxel_ptr: torch.Tensor,
    dense_point_features_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points_ptr: input points tensor, shape (num_points, num_features_per_point)
        voxelization parameters
        dense_num_points_per_voxel_ptr: output counter for number of points in each dense voxel, shape
        (max_num_voxels), tensor must be filled with 0 before calling this kernel.
        dense_point_features_ptr: output point features ordered by dense voxels, shape
        (max_num_voxels, max_num_points_per_voxel, num_features_per_point), tensor must be filled with 0 before
        calling this kernel.
    """
    block_idx = tl.program_id(axis=0)
    point_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    point_mask = point_idx < num_points

    point_x = tl.load(points_ptr + 4 * point_idx + 0, mask=point_mask, other=max_x + voxel_dim_x)
    point_y = tl.load(points_ptr + 4 * point_idx + 1, mask=point_mask, other=max_y + voxel_dim_y)
    point_z = tl.load(points_ptr + 4 * point_idx + 2, mask=point_mask, other=max_z + voxel_dim_z)
    point_w = tl.load(points_ptr + 4 * point_idx + 3, mask=point_mask, other=0)

    voxel_x = tl.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)

    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = point_mask & valid_x & valid_y & valid_z

    flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x
    point_idx_in_voxel = tl.atomic_add(dense_num_points_per_voxel_ptr + flat_voxel_idx, 1, mask=valid_point)

    output_idx = flat_voxel_idx * max_num_points_per_voxel + point_idx_in_voxel
    output_mask = valid_point & (point_idx_in_voxel < max_num_points_per_voxel)
    tl.store(dense_point_features_ptr + output_idx * 4 + 0, point_x, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 1, point_y, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 2, point_z, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 3, point_w, mask=output_mask)


@triton.jit
def generate_voxels_kernel(  # noqa: PLR0913, D417
    # input
    dense_point_features_ptr: torch.Tensor,
    dense_num_points_per_voxel_ptr: torch.Tensor,
    # parameters
    grid_dim_x: int,
    grid_dim_y: int,
    max_num_points_per_voxel: int,
    max_num_voxels: int,
    # output
    num_filled_voxels_ptr: torch.Tensor,
    num_points_per_voxel_ptr: torch.Tensor,
    point_features_ptr: torch.Tensor,
    voxel_indices_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Convert dense voxels into sparse/contiguous non-empty voxels.

    Args:
        dense_point_features_ptr: input point features tensor, shape
        (max_num_voxels, max_num_points_per_voxel, num_features_per_point)
        dense_num_points_per_voxel_ptr: input counter for number of points in each dense voxel, shape
        (max_num_voxels)
        voxelization parameters
        num_filled_voxels_ptr: output counter for the total number of non-empty voxels
        num_points_per_voxel_ptr: output counter for the number of points in each filled voxel, capped to
        max_num_points_per_voxel, shape (num_filled_voxels)
        point_features_ptr: output point features for each filled voxel, shape
        (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        voxel_indices_ptr: output per voxel 3D coordinates tensor, shape
        (num_filled_voxels, 4), only first 3 fields are set for indices in x, y, z
    """
    pid = tl.program_id(0)
    flat_voxel_idx = pid * cxpr_block_size + tl.arange(0, cxpr_block_size)

    num_points_in_voxel = tl.load(
        dense_num_points_per_voxel_ptr + flat_voxel_idx, mask=flat_voxel_idx < max_num_voxels, other=0
    )

    num_points_in_voxel = tl.minimum(num_points_in_voxel, max_num_points_per_voxel)
    valid_voxel = num_points_in_voxel > 0

    voxel_idx = tl.atomic_add(num_filled_voxels_ptr + tl.zeros_like(valid_voxel), 1, mask=valid_voxel)

    # store num_points_per_voxel with clipping
    tl.store(num_points_per_voxel_ptr + voxel_idx, num_points_in_voxel, mask=valid_voxel)

    # convert flat voxel index to 3d coordinates
    voxel_x = flat_voxel_idx % grid_dim_x
    voxel_y = (flat_voxel_idx // grid_dim_x) % grid_dim_y
    voxel_z = flat_voxel_idx // (grid_dim_y * grid_dim_x)

    # store 3d indices
    # index is padded to int4 type same as cuda impl
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 0, voxel_x, mask=valid_voxel)
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 1, voxel_y, mask=valid_voxel)
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 2, voxel_z, mask=valid_voxel)

    # store all feature points, including padded 0s
    for point_idx in range(0, max_num_points_per_voxel, 1):
        input_idx = flat_voxel_idx * max_num_points_per_voxel + point_idx
        point_x = tl.load(dense_point_features_ptr + input_idx * 4 + 0, mask=valid_voxel)
        point_y = tl.load(dense_point_features_ptr + input_idx * 4 + 1, mask=valid_voxel)
        point_z = tl.load(dense_point_features_ptr + input_idx * 4 + 2, mask=valid_voxel)
        point_w = tl.load(dense_point_features_ptr + input_idx * 4 + 3, mask=valid_voxel)

        output_idx = voxel_idx * max_num_points_per_voxel + point_idx
        tl.store(point_features_ptr + output_idx * 4 + 0, point_x, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 1, point_y, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 2, point_z, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 3, point_w, mask=valid_voxel)


@triton.jit
def collect_point_features_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_features_per_point: int,
    segment_offsets_ptr: torch.Tensor,
    num_filled_voxels: int,
    point_indices_ptr: torch.Tensor,
    # parameters
    max_num_points_per_voxel: int,
    # output
    point_features_ptr: torch.Tensor,
    capped_num_points_per_voxel_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points_ptr: input points tensor, shape (num_points, num_features_per_point)
        segment_offsets_ptr: input segment end offsets, shape (num_filled_voxels)
        point_indices_ptr: input raw point indices, shape (num_valid_points)
        voxelization parameters
        point_features_ptr: output voxel point features, shape (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        capped_num_points_per_voxel_ptr: output number of points per voxel tensor after capping, shape (num_filled_voxels)
    """
    block_idx = tl.program_id(axis=0)
    voxel_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    voxel_mask = voxel_idx < num_filled_voxels

    # top n filtering
    segment_start = tl.load(segment_offsets_ptr + voxel_idx - 1, mask=(voxel_mask & (voxel_idx > 0)), other=0)
    segment_end = tl.load(segment_offsets_ptr + voxel_idx, mask=voxel_mask, other=0)
    num_points_in_voxel = segment_end - segment_start
    num_points_in_voxel = tl.minimum(num_points_in_voxel, max_num_points_per_voxel)
    tl.store(capped_num_points_per_voxel_ptr + voxel_idx, num_points_in_voxel, mask=voxel_mask)

    for voxel_point_idx in range(0, max_num_points_per_voxel, 1):
        # this mask is sufficient since other num_points_in_voxel == 0
        per_voxel_mask = voxel_point_idx < num_points_in_voxel

        raw_point_idx = tl.load(point_indices_ptr + segment_start + voxel_point_idx, mask=per_voxel_mask)
        output_idx = voxel_idx * max_num_points_per_voxel + voxel_point_idx
        for feature_idx in range(0, num_features_per_point, 1):
            value = tl.load(
                points_ptr + raw_point_idx * num_features_per_point + feature_idx, mask=per_voxel_mask, other=0
            )
            tl.store(point_features_ptr + output_idx * num_features_per_point + feature_idx, value, mask=voxel_mask)
