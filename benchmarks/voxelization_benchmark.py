# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization benchmark."""


from typing import Final
import torch

#from conch_cuda_ext.ops.vision.voxelization.voxelization import generate_voxels as generate_voxels_cuda
from conch.ops.vision.voxelization import (
        VoxelizationParameter, generate_voxels, voxelization_stable, collect_point_features,)

# kernels parallelize on either num_points or max/actual number of voxels
# change num_points and range to adjust point filtering & grid occupancy
# keep max_num_points_per_voxel less than 64 for ideal performance for this impl
# cuda impl requires num_features_per_point = 4
def main() -> None:
    """Benchmark voxelization."""

    device: Final = torch.device("cuda")
    torch.set_default_device(device)

    use_triton = True
    # use 500k points at minimum for a10
    num_points = 500000
    num_features_per_point = 4
    range_xyz = 50.0
    points = torch.randn((num_points, num_features_per_point), device=device) * range_xyz

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

    # cuda version
#    _ = generate_voxels_cuda(points,
#            param.min_range[0],
#            param.min_range[1],
#            param.min_range[2],
#            param.max_range[0],
#            param.max_range[1],
#            param.max_range[2],
#            param.voxel_dim[0],
#            param.voxel_dim[1],
#            param.voxel_dim[2],
#            param.grid_dim[0],
#            param.grid_dim[1],
#            param.max_num_points_per_voxel)

    # triton/torch hybrid, 2-step, stable voxelization first then generate a feature tensor the same format as above
    actual_num_points_per_voxel, point_raw_indices, flat_voxel_indices_stable = voxelization_stable(
        points, param, use_triton=use_triton
    )
    point_features_stable, capped_num_points_per_voxel = collect_point_features(
        points, actual_num_points_per_voxel, point_raw_indices, param, use_triton=use_triton
    )

    print("voxelization done, stats:")
    print(f"number of filled voxels: {num_points_per_voxel.shape[0]}")
    print(f"Avg number of points per voxel: {torch.mean(actual_num_points_per_voxel, dtype=torch.float32)}")
    print(f"Max number of points per voxel: {torch.max(actual_num_points_per_voxel)}")
    overflow_count = (actual_num_points_per_voxel > param.max_num_points_per_voxel).sum()
    print(f"Number of voxels with overflowing points: {overflow_count}")

if __name__ == "__main__":
    main()
