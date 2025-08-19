# pyright: reportPrivateUsage=false
"""Test voxelization."""

import pytest
import torch

from conch.ops.vision.voxelization import (
        VoxelizationParameter, generate_voxels, voxelization_stable, collect_point_features,)

def voxel_coords_to_flat_indices(coords: torch.Tensor, grid_dim: tuple[int, int, int]) -> torch.Tensor:
    """Convert 3d voxel coordinates to flat indices."""
    voxel_x = coords[:, 0]
    voxel_y = coords[:, 1]
    voxel_z = coords[:, 2]
    grid_dim_x = grid_dim[0]
    grid_dim_y = grid_dim[1]
    return (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x


# wether or not use Triton for the reference Torch impl
@pytest.mark.parametrize("use_triton", [True, False])
def test_voxelization(use_triton: bool) -> None:
    """Test triton/pytorch voxelization."""
    num_points = 500000
    num_features_per_point = 4
    range_xyz = 50.0
    points = torch.randn((num_points, num_features_per_point), device="cuda") * range_xyz

    param = VoxelizationParameter(
        min_range=(-range_xyz, -range_xyz, -range_xyz),
        max_range=(range_xyz, range_xyz, range_xyz),
        voxel_dim=(2.5, 2.5, 2.5),
        max_num_points_per_voxel=4,
    )

    print(f"Grid dimensions: {param.grid_dim}")
    print(f"Max number of voxels: {param.max_num_voxels}")
    print(f"Max number of points per voxel: {param.max_num_points_per_voxel}")

    # pure triton version
    num_points_per_voxel, point_features, voxel_indices = generate_voxels(points, param)

    # triton/torch hybrid, 2-step, stable voxelization first then generate a feature tensor the same format as above
    actual_num_points_per_voxel, point_raw_indices, flat_voxel_indices_stable = voxelization_stable(
        points, param, use_triton=use_triton
    )
    point_features_stable, capped_num_points_per_voxel = collect_point_features(
        points, actual_num_points_per_voxel, point_raw_indices, param, use_triton=use_triton
    )

    print("voxelization done, stats:")
    print(f"Avg number of points per voxel: {torch.mean(actual_num_points_per_voxel, dtype=torch.float32)}")
    print(f"Max number of points per voxel: {torch.max(actual_num_points_per_voxel)}")
    overflow_count = (actual_num_points_per_voxel > param.max_num_points_per_voxel).sum()
    print(f"Number of voxels with overflowing points: {overflow_count}")

    assert point_raw_indices.size(dim=0) <= num_points
    assert flat_voxel_indices_stable.shape == actual_num_points_per_voxel.shape
    assert flat_voxel_indices_stable.shape == num_points_per_voxel.shape
    num_filled_voxels = num_points_per_voxel.size(dim=0)
    assert point_features.shape == (num_filled_voxels, param.max_num_points_per_voxel, num_features_per_point)
    assert point_features_stable.shape == (num_filled_voxels, param.max_num_points_per_voxel, num_features_per_point)
    assert voxel_indices.shape == (num_filled_voxels, 4)

    # for nondeterministic output, sort voxels by row major flat indices
    flat_voxel_indices = voxel_coords_to_flat_indices(voxel_indices, param.grid_dim)
    flat_voxel_indices_sorted, permute_indices = torch.sort(flat_voxel_indices, stable=True)

    # match voxel indices
    assert torch.allclose(flat_voxel_indices_stable, flat_voxel_indices_sorted)

    # match per voxel point counters
    sorted_num_points_per_voxel = num_points_per_voxel[permute_indices]
    actual_num_points_per_voxel.clamp_(max=param.max_num_points_per_voxel)
    assert torch.allclose(sorted_num_points_per_voxel, actual_num_points_per_voxel)
    assert torch.allclose(sorted_num_points_per_voxel, capped_num_points_per_voxel)

    # match per voxel features
    point_features_sorted = point_features[permute_indices]
    voxel_mean_x = torch.mean(point_features_sorted[:, :, 0], dim=1)
    voxel_mean_y = torch.mean(point_features_sorted[:, :, 1], dim=1)
    voxel_mean_z = torch.mean(point_features_sorted[:, :, 2], dim=1)
    voxel_mean_x_stable = torch.mean(point_features_stable[:, :, 0], dim=1)
    voxel_mean_y_stable = torch.mean(point_features_stable[:, :, 1], dim=1)
    voxel_mean_z_stable = torch.mean(point_features_stable[:, :, 2], dim=1)
    voxel_dim_x = param.voxel_dim[0]
    voxel_dim_y = param.voxel_dim[1]
    voxel_dim_z = param.voxel_dim[2]
    assert torch.allclose(voxel_mean_x, voxel_mean_x_stable, atol=voxel_dim_x)
    assert torch.allclose(voxel_mean_y, voxel_mean_y_stable, atol=voxel_dim_y)
    assert torch.allclose(voxel_mean_z, voxel_mean_z_stable, atol=voxel_dim_z)
