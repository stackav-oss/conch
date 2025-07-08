# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization PyTorch reference implementation."""

from importlib.util import find_spec

import torch

from conch import envs


def _voxelization_pytorch(
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert kitti points(N, >=3) to voxels.

    Args:
        points (torch.Tensor): [N, ndim]. Points[:, :3] contain xyz points
            and points[:, 3:] contain other information like reflectivity.
        voxel_size: The size of each voxel, shape: [3].
        coordinate_range: The coordinate range of voxel, shape [6].
        max_points (int, optional): maximum points contained in a voxel. if
            max_points=-1, it means using dynamic_voxelize. Default: 35.
        max_voxels (int, optional): maximum voxels this function create.
            for second, 20000 is a good choice. Users should shuffle points
            before call this function because max_voxels may drop points.
            Default: 20000.

    Returns:
        tuple[torch.Tensor]: tuple[torch.Tensor]: A tuple contains three
        elements. The first one is the output voxels with the shape of
        [M, max_points, n_dim], which only contain points and returned
        when max_points != -1. The second is the voxel coordinates with
        shape of [M, 3]. The last is number of point per voxel with the
        shape of [M], which only returned when max_points != -1.
    """
    points_xyz = points[:, :3]
    voxel_size_x, voxel_size_y, voxel_size_z = voxel_size
    (
        coordinate_range_x_min,
        coordinate_range_y_min,
        coordinate_range_z_min,
        coordinate_range_x_max,
        coordinate_range_y_max,
        coordinate_range_z_max,
    ) = coordinate_range

    num_points, _ = points_xyz.shape
    assert num_points > 0, "points should not be empty"

    # Calculate voxel indices
    voxel_indices_x = ((points_xyz[:, 0] - coordinate_range_x_min) / voxel_size_x).floor().long()
    voxel_indices_y = ((points_xyz[:, 1] - coordinate_range_y_min) / voxel_size_y).floor().long()
    voxel_indices_z = ((points_xyz[:, 2] - coordinate_range_z_min) / voxel_size_z).floor().long()
    voxel_indices = torch.stack((voxel_indices_x, voxel_indices_y, voxel_indices_z), dim=1)

    # Calculate voxel coordinates
    voxel_coordinates = voxel_indices - torch.tensor(
        [
            int(coordinate_range_x_min // voxel_size_x),
            int(coordinate_range_y_min // voxel_size_y),
            int(coordinate_range_z_min // voxel_size_z),
        ],
        device=points.device,
    )

    # Calculate voxel indices in a flat format
    voxel_flat_indices = (
        voxel_coordinates[:, 0] * int((coordinate_range_y_max - coordinate_range_y_min) / voxel_size_y)
        + voxel_coordinates[:, 1] * int((coordinate_range_z_max - coordinate_range_z_min) / voxel_size_z)
        + voxel_coordinates[:, 2]
    )

    # return voxel_flat_indices

    # Sort points by voxel indices
    sorted_indices = torch.argsort(voxel_flat_indices)
    sorted_voxel_indices = voxel_flat_indices[sorted_indices]
    sorted_points = points[sorted_indices]

    # Group points by voxel indices
    unique_voxel_indices, inverse_indices = torch.unique(sorted_voxel_indices, return_inverse=True)
    num_voxels = unique_voxel_indices.size(0)

    if num_voxels > max_voxels:
        # If the number of voxels exceeds max_voxels, randomly select max_voxels
        selected_indices = torch.randperm(num_voxels)[:max_voxels]
        unique_voxel_indices = unique_voxel_indices[selected_indices]
        inverse_indices = inverse_indices[sorted_indices][selected_indices]

    # Create output voxels
    if max_points_per_voxel > 0:
        voxels = torch.zeros(
            (num_voxels, max_points_per_voxel, points.size(1)), dtype=points.dtype, device=points.device
        )
        num_points_per_voxel = torch.zeros(num_voxels, dtype=torch.int32, device=points.device)

        for i in range(num_voxels):
            voxel_mask = inverse_indices == i
            points_in_voxel = sorted_points[voxel_mask]
            num_points_in_voxel = points_in_voxel.size(0)

            if num_points_in_voxel > max_points_per_voxel:
                points_in_voxel = points_in_voxel[:max_points_per_voxel]
                num_points_in_voxel = max_points_per_voxel

            voxels[i, :num_points_in_voxel] = points_in_voxel
            num_points_per_voxel[i] = num_points_in_voxel

        # Collect voxel coordinates for each unique voxel
        voxel_coords_out = torch.zeros((num_voxels, 3), dtype=torch.long, device=points.device)
        for i in range(num_voxels):
            voxel_mask = inverse_indices == i
            if voxel_mask.any():
                # Get the first point in this voxel and extract its voxel coordinates
                first_point_idx = torch.where(voxel_mask)[0][0]
                voxel_coords_out[i] = voxel_coordinates[sorted_indices[first_point_idx]]

        return voxels, voxel_coords_out, num_points_per_voxel
    else:
        # If max_points_per_voxel is -1, return dynamic voxelization
        voxels = sorted_points.view(-1, points.size(1))
        # For dynamic mode, return coordinates for each point
        return voxels, voxel_coordinates, torch.ones(num_voxels, dtype=torch.int32, device=points.device)


def _voxelization_mmcv(
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from conch_cuda_ext.ops.vision.voxelization.voxelization import dynamic_voxelize_forward, hard_voxelize_forward

    if max_points_per_voxel == -1 or max_voxels == -1:
        coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
        dynamic_voxelize_forward(points, coors, list(voxel_size), list(coordinate_range), 3)
        return coors
    else:
        voxels = points.new_zeros(size=(max_voxels, max_points_per_voxel, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
        voxel_num = hard_voxelize_forward(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            list(voxel_size),
            list(coordinate_range),
            max_points_per_voxel,
            max_voxels,
            3,
        )
        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


def voxelization(
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if envs.CONCH_ENABLE_CUDA_EXT:
        if find_spec("conch_cuda_ext") is None:
            raise ImportError("Conch CUDA extension is not available. Please build the extension first.")

        return _voxelization_mmcv(
            points,
            voxel_size,
            coordinate_range,
            max_points_per_voxel,
            max_voxels,
        )

    return _voxelization_pytorch(
        points,
        voxel_size,
        coordinate_range,
        max_points_per_voxel,
        max_voxels,
    )
