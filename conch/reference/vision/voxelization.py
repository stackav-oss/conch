# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton point cloud voxeliation."""

import torch
import triton
import triton.language as tl

from conch.ops.vision.voxelization import VoxelizationParameter


@triton.jit  # type: ignore[misc]
def filter_and_label_points_triton_kernel(
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


def filter_and_label_points_torch(
    points: torch.Tensor,
    min_range: tuple[float, float, float],
    voxel_dim: tuple[float, float, float],
    grid_dim: tuple[int, int, int],
    max_num_voxels: int,
    point_voxel_indices: torch.Tensor,
) -> None:
    """Filter valid points and label each with a flat voxel index.

    Args:
        points: Input points with shape (num_points, num_features_per_point), first three fields of each point should be
        x, y, z.
        min_range: Minimum bounds (min_x, min_y, min_z).
        max_range: Maximum bounds (max_x, max_y, max_z).
        voxel_dim: Voxel dimensions (voxel_dim_x, voxel_dim_y, voxel_dim_z).
        grid_dim: Grid dimensions (grid_dim_x, grid_dim_y, grid_dim_z).
        point_voxel_indices: Output flat voxel indices for each point.
    """
    point_x = points[:, 0]
    point_y = points[:, 1]
    point_z = points[:, 2]

    min_x, min_y, min_z = min_range
    voxel_dim_x, voxel_dim_y, voxel_dim_z = voxel_dim
    grid_dim_x, grid_dim_y, grid_dim_z = grid_dim

    # Compute voxel indices
    voxel_x = torch.floor((point_x - min_x) / voxel_dim_x).to(torch.int32)
    voxel_y = torch.floor((point_y - min_y) / voxel_dim_y).to(torch.int32)
    voxel_z = torch.floor((point_z - min_z) / voxel_dim_z).to(torch.int32)

    # bounds check on voxel indices
    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = valid_x & valid_y & valid_z

    flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x
    point_voxel_indices[:] = torch.where(valid_point, flat_voxel_idx, max_num_voxels)


def voxelization_stable(
    points: torch.Tensor, param: VoxelizationParameter, use_triton: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize input points. output is deterministic running back to back with the same input.

    Args:
        points: input points; expected dimensions (num_points, num_features_per_point), first three fields of each point
        should be x, y, z.
        param: voxelization parameters.
        use_triton: whether to use a Triton kernel for labeling points

    Returns:
        tuple, voxels SoA sorted by flat voxel indices on the grid:
            num_points_per_voxel, shape [num_filled_voxels], note this is actual number of points in each voxel without clipping.
            point_indices: original point indices grouped by voxels, shape [num_valid_points], points within the same voxel are
            contiguous with segment size specified in num_points_per_voxel.
            flat_voxel_indices, shape [num_filled_voxels].
    """
    device = points.device
    num_points, num_features_per_point = points.shape

    # init raw indices
    point_raw_indices = torch.arange(num_points, device=device)
    point_voxel_indices = torch.empty((num_points,), dtype=torch.int32, device=device)

    if use_triton:
        # compute point wise flat voxel indices
        block_size = 256
        num_threads_per_warp = 32
        grid = (triton.cdiv(num_points, block_size),)
        filter_and_label_points_triton_kernel[grid](
            points,
            num_points,
            num_features_per_point,
            param.min_range[0],
            param.min_range[1],
            param.min_range[2],
            param.max_range[0],
            param.max_range[1],
            param.max_range[2],
            param.voxel_dim[0],
            param.voxel_dim[1],
            param.voxel_dim[2],
            param.grid_dim[0],
            param.grid_dim[1],
            param.grid_dim[2],
            param.max_num_voxels,
            point_voxel_indices,
            cxpr_block_size=block_size,
            num_warps=block_size // num_threads_per_warp,
        )
    else:
        filter_and_label_points_torch(
            points, param.min_range, param.voxel_dim, param.grid_dim, param.max_num_voxels, point_voxel_indices
        )

    # mask for points within bound, can skip select step when most points are valid
    mask = point_voxel_indices < param.max_num_voxels
    # value
    raw_indices_selected = torch.masked_select(point_raw_indices, mask)
    # key
    voxel_indices_selected = torch.masked_select(point_voxel_indices, mask)

    # group points into voxels with sort_by_key(), use stable to keep original points ordering
    sorted_voxel_indices, permute_indices = torch.sort(voxel_indices_selected, stable=True)
    sorted_raw_indices = raw_indices_selected[permute_indices]
    # run length encode
    voxel_indices, num_points_per_voxel = torch.unique_consecutive(sorted_voxel_indices, return_counts=True)

    return num_points_per_voxel.to(torch.int32), sorted_raw_indices, voxel_indices


@triton.jit  # type: ignore[misc]
def collect_point_features_triton_kernel(
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

    for voxel_point_idx in range(0, max_num_points_per_voxel):
        # this mask is sufficient since other num_points_in_voxel == 0
        per_voxel_mask = voxel_point_idx < num_points_in_voxel

        raw_point_idx = tl.load(point_indices_ptr + segment_start + voxel_point_idx, mask=per_voxel_mask)
        output_idx = voxel_idx * max_num_points_per_voxel + voxel_point_idx
        for feature_idx in range(0, num_features_per_point):
            value = tl.load(
                points_ptr + raw_point_idx * num_features_per_point + feature_idx, mask=per_voxel_mask, other=0
            )
            tl.store(point_features_ptr + output_idx * num_features_per_point + feature_idx, value, mask=voxel_mask)


def collect_point_features_torch(
    points: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    segment_offsets: torch.Tensor,
    point_indices: torch.Tensor,
    max_num_points_per_voxel: int,
    point_features: torch.Tensor,
    capped_num_points_per_voxel: torch.Tensor,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points: input points tensor, shape (num_points, num_features_per_point)
        num_points_per_voxel: input number of points per voxel tensor, shape (num_filled_voxels)
        segment_offsets: input segment end offsets, shape (num_filled_voxels)
        point_indices: input raw point indices, shape (num_valid_points)
        voxelization parameters
        point_features: output voxel point features, shape (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        capped_num_points_per_voxel: output number of points per voxel tensor after capping, shape (num_filled_voxels)
    """
    # inclusive sum to exclusive sum
    start_indices = torch.cat(
        (torch.zeros(1, dtype=segment_offsets.dtype, device=segment_offsets.device), segment_offsets[:-1])
    )
    capped_num_points_per_voxel[:] = torch.where(
        num_points_per_voxel > max_num_points_per_voxel, max_num_points_per_voxel, num_points_per_voxel
    )
    # init feature tensor with 0 first
    point_features.zero_()

    # top n filtering
    for voxel_idx, (start_idx, num_points_in_voxel) in enumerate(
        zip(start_indices, capped_num_points_per_voxel, strict=False)
    ):
        raw_indices = point_indices[start_idx : start_idx + num_points_in_voxel]
        for point_idx_in_voxel, raw_point_idx in enumerate(raw_indices):
            point_features[voxel_idx, point_idx_in_voxel, :] = points[raw_point_idx, :]


def collect_point_features(
    points: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    point_indices: torch.Tensor,
    param: VoxelizationParameter,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather voxel point features from raw points and voxelization result.

    Args:
        points: input points, expected dimensions (num_points, num_features_per_point).
        num_points_per_voxel: shape [num_filled_voxels], actual number of points in each voxel.
        point_indices: original point indices grouped by voxels, shape [num_valid_points]
        param: voxelization parameters.
        use_triton: whether to use a Triton kernel

    Returns:
        point_features: shape [num_valid_points, max_num_points_per_voxel, num_features_per_point], empty points are
        filled with 0.
        capped_num_points_per_voxel: shape [num_filled_voxels], number of points in each voxel after max capping.
    """
    device = points.device
    num_points, num_features_per_point = points.shape

    (num_filled_voxels,) = num_points_per_voxel.shape
    assert num_filled_voxels <= param.max_num_voxels
    (num_valid_points,) = point_indices.shape
    assert num_valid_points <= num_points

    segment_offsets = torch.cumsum(num_points_per_voxel, dim=0)

    # output
    capped_num_points_per_voxel = torch.empty_like(num_points_per_voxel)
    point_features = torch.empty(
        (num_filled_voxels, param.max_num_points_per_voxel, num_features_per_point), dtype=torch.float, device=device
    )

    if use_triton:
        # one thread per voxel, ideally max_num_points_per_voxel < 64
        block_size = 256
        num_threads_per_warp = 32
        grid = (triton.cdiv(num_filled_voxels, block_size),)
        collect_point_features_triton_kernel[grid](
            points,
            num_features_per_point,
            segment_offsets,
            num_filled_voxels,
            point_indices,
            param.max_num_points_per_voxel,
            point_features,
            capped_num_points_per_voxel,
            cxpr_block_size=block_size,
            num_warps=block_size // num_threads_per_warp,
        )
    else:
        collect_point_features_torch(
            points,
            num_points_per_voxel,
            segment_offsets,
            point_indices,
            param.max_num_points_per_voxel,
            point_features,
            capped_num_points_per_voxel,
        )

    return point_features, capped_num_points_per_voxel
