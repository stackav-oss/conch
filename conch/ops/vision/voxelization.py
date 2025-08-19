# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton point cloud voxeliation."""

from dataclasses import dataclass

from conch.kernels.vision.voxelization import (
generate_dense_voxels_kernel,
generate_voxels_kernel,
filter_and_label_points_kernel,
collect_point_features_kernel,
)

import triton
import torch

@dataclass
class VoxelizationParameter:
    """Parameters."""

    min_range: tuple[float, float, float]
    max_range: tuple[float, float, float]
    voxel_dim: tuple[float, float, float]
    grid_dim: tuple[int, int, int]
    max_num_points_per_voxel: int
    max_num_voxels: int

    def __init__(
        self,
        min_range: tuple[float, float, float],
        max_range: tuple[float, float, float],
        voxel_dim: tuple[float, float, float],
        max_num_points_per_voxel: int,
    ) -> None:
        """Init parameters."""
        self.min_range = min_range
        self.max_range = max_range
        self.voxel_dim = voxel_dim
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.grid_dim = self._compute_grid_dim()
        self.max_num_voxels = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

    def _compute_grid_dim(self) -> tuple[int, int, int]:
        """Compute grid dimensions."""
        grid_x = round((self.max_range[0] - self.min_range[0]) / self.voxel_dim[0])
        grid_y = round((self.max_range[1] - self.min_range[1]) / self.voxel_dim[1])
        grid_z = round((self.max_range[2] - self.min_range[2]) / self.voxel_dim[2])
        return (grid_x, grid_y, grid_z)

def generate_voxels(
    points: torch.Tensor, param: VoxelizationParameter
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates voxels from input points, output voxels and points are randomly ordered due to use of atomics.

    Args:
        points: input points; expected dimensions (num_points, 4), last dimension should have fields of x,y,z,_.
        param: parameters.

    Returns:
        tuple of voxels SoA:
            capped_num_points_per_voxel, shape [num_filled_voxels], note per voxel point counters are capped with max_num_points_per_voxel.
            point_features, shape [num_filled_voxels, max_num_points_per_voxel, num_features_per_point],
            empty points are filled with 0.
            voxel_indices, shape [num_filled_voxels, 4], only first 3 fields are used for x,y,z indices.
    """
    assert points.is_cuda
    device = points.device
    num_points, num_features_per_point = points.shape
    assert num_features_per_point == 4  # noqa: PLR2004
    # same as original nvidia cuda impl
    num_elements_per_voxel_index = 4

    # dense (must set to 0s)
    dense_num_points_per_voxel = torch.zeros((param.max_num_voxels), dtype=torch.int32, device=device)
    dense_point_features = torch.zeros(
        (param.max_num_voxels, param.max_num_points_per_voxel, num_features_per_point), dtype=torch.float, device=device
    )

    # sparse/contiguous output
    num_filled_voxels = torch.zeros((1), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.empty_like(dense_num_points_per_voxel)
    point_features = torch.empty_like(dense_point_features)
    voxel_indices = torch.empty((param.max_num_voxels, num_elements_per_voxel_index), dtype=torch.int32, device=device)

    block_size = 256
    num_threads_per_warp = 32

    # first generate dense voxels
    grid = (triton.cdiv(num_points, block_size),)
    generate_dense_voxels_kernel[grid](
        points,
        num_points,
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
        param.max_num_points_per_voxel,
        dense_num_points_per_voxel,
        dense_point_features,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
    )

    # compress into contiguous/sparse filled voxels
    grid = (triton.cdiv(param.max_num_voxels, block_size),)
    generate_voxels_kernel[grid](
        dense_point_features,
        dense_num_points_per_voxel,
        param.grid_dim[0],
        param.grid_dim[1],
        param.max_num_points_per_voxel,
        param.max_num_voxels,
        num_filled_voxels,
        num_points_per_voxel,
        point_features,
        voxel_indices,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
    )

    total_filled_voxels = num_filled_voxels.cpu()[0]

    return (
        num_points_per_voxel[:total_filled_voxels],
        point_features[:total_filled_voxels, :, :],
        voxel_indices[:total_filled_voxels, :],
    )

def filter_and_label_points_torch(  # noqa: PLR0913, D417
    points: torch.Tensor,
    min_range: tuple[float, float, float],
    voxel_dim: tuple[float, float, float],
    grid_dim: tuple[int, int, int],
    max_num_voxels: int,
    point_voxel_indices: torch.Tensor,
) -> None:
    """Filter valid points and label each with a flat voxel index.

    Args:
        points: Input points with shape (num_points, num_features_per_point).
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
        points: input points; expected dimensions (num_points, num_features_per_point), first three fields of last
        dimentions should be x, y, z.
        param: voxelization parameters.
        use_triton: whether to use a Triton kernel for labeling points

    Returns:
        tuple, voxels SoA sorted by flat voxel indices on the grid:
            num_points_per_voxel, shape [num_filled_voxels], note this is actual number of points in each voxel without clipping.
            point_indices: original point indices grouped by voxels, shape [num_valid_points], points within the same voxel are
            contiguous with segment size specified in num_points_per_voxel.
            flat_voxel_indices, shape [num_filled_voxels].
    """
    assert points.is_cuda
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
        filter_and_label_points_kernel[grid](
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
            num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
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


def collect_point_features_torch(  # noqa: PLR0913, D417
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
    """Gather voxel point features from original points and voxelization result.

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
    assert points.is_cuda
    device = points.device
    num_points, num_features_per_point = points.shape
    assert num_features_per_point == 4  # noqa: PLR2004

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
        # one thread per voxel, when max_num_points_per_voxel is larger than 64, use one block per voxel
        block_size = 256
        num_threads_per_warp = 32
        grid = (triton.cdiv(num_filled_voxels, block_size),)
        collect_point_features_kernel[grid](
            points,
            num_features_per_point,
            segment_offsets,
            num_filled_voxels,
            point_indices,
            param.max_num_points_per_voxel,
            point_features,
            capped_num_points_per_voxel,
            cxpr_block_size=block_size,
            num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
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
