# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization implementation."""

import torch

from conch.kernels.vision.voxelization import voxelization_launcher


def voxelization(
    points: torch.Tensor,
    voxel_size: tuple[int, int, int],
    coordinate_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int = 35,
    max_voxels: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert points to voxel coordinates.

    Args:
        points: The points to convert, shape: (N, ndim). Points[:, :3] contain xyz points
            and points[:, 3:] contain other information like reflectivity.
        voxel_size: The size of each voxel, shape: (3).
        coordinate_range: The coordinate range of voxel, shape: (6).
            The order is (x_min, y_min, z_min, x_max, y_max, z_max).

    Returns:
    """
    num_points, num_features = points.shape

    num_points_per_voxel = torch.zeros((max_voxels,), dtype=torch.int32, device="cuda")
    point_features = torch.empty((max_voxels, max_points_per_voxel, num_features), dtype=points.dtype, device=points.device)

    voxelization_launcher(
        num_points_per_voxel=num_points_per_voxel,
        point_features=point_features,
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    return num_points_per_voxel, point_features
