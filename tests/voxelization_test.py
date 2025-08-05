# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test voxelization implementations."""

from typing import Final

import pytest
import torch

from conch.platforms import current_platform

# from conch.reference.vision.voxelization import _voxelization_mmcv, _voxelization_pytorch
from conch.reference.vision.voxelization import _dynamic_voxelize, _dynamic_voxelize_mmcv, _voxelization_mmcv
from conch.ops.vision.voxelization import voxelization as voxelization_conch
from conch.third_party.vllm.utils import seed_everything

_DTYPES: Final = [torch.float32, torch.float16]
_NUM_POINTS: Final = [100, 500, 1000]
_VOXEL_SIZES: Final = [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.5, 0.5, 0.5)]
_MAX_POINTS_PER_VOXEL: Final = [10, 35, 50]
_MAX_VOXELS: Final = [1000, 5000, 10000]


def _create_test_points(
    num_points: int,
    coordinate_range: tuple[float, float, float, float, float, float],
    num_features: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create test point cloud data within the specified coordinate range."""
    x_min, y_min, z_min, x_max, y_max, z_max = coordinate_range

    # Generate points within the coordinate range
    points = torch.zeros(num_points, num_features, dtype=dtype, device=device)

    # XYZ coordinates
    points[:, 0] = torch.rand(num_points, dtype=dtype, device=device) * (x_max - x_min) + x_min
    points[:, 1] = torch.rand(num_points, dtype=dtype, device=device) * (y_max - y_min) + y_min
    points[:, 2] = torch.rand(num_points, dtype=dtype, device=device) * (z_max - z_min) + z_min

    # Additional features (e.g., reflectivity)
    if (num_extra_features := num_features - 3) > 0:
        points[:, 3:] = torch.rand(num_points, num_extra_features, dtype=dtype, device=device)

    return points


@pytest.mark.parametrize("max_points_per_voxel", [-1])
@pytest.mark.parametrize("seed", range(2))
# @pytest.mark.parametrize("num_features", [3, 4, 63, 64])
@pytest.mark.parametrize("num_features", [4, 63, 64])
def test_dynamic_voxelization(max_points_per_voxel: int, seed: int, num_features: int) -> None:
    """Test dynamic voxelization."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    num_points = 500
    # x, y, z
    voxel_size = (0.2, 0.2, 0.2)
    # x_min, y_min, z_min, x_max, y_max, z_max
    coordinate_range = (-5.0, -5.0, -2.0, 5.0, 5.0, 2.0)
    # max_voxels = -1
    _ = max_points_per_voxel

    # Create test points
    points = _create_test_points(num_points, coordinate_range, num_features=num_features, dtype=dtype, device=device)

    # Test PyTorch implementation
    # voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
    voxels_pytorch = _dynamic_voxelize(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        # max_points_per_voxel=max_points_per_voxel,
    )

    # points_cuda = points.float()

    # coors_cuda = _voxelization_mmcv(
    #     points=points_cuda,
    #     voxel_size=voxel_size,
    #     coordinate_range=coordinate_range,
    #     max_points_per_voxel=max_points_per_voxel,
    #     max_voxels=max_voxels,
    # )
    coors_cuda = _dynamic_voxelize_mmcv(
        # points=points_cuda,
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
    )

    # Note: for out-of-bounds points, the CUDA implementation may return empty coordinates
    # while the PyTorch implementation will return (-1, -1, -1) for those points.
    torch.testing.assert_close(coors_cuda, voxels_pytorch, rtol=1e-5, atol=1e-5)

    conch_num_points_per_voxel, conch_point_features = voxelization_conch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
    )

    # print(f"{conch_num_points_per_voxel = }")
    # print(f"{conch_point_features = }")

    # assert False


@pytest.mark.parametrize("num_points", [5, 5000, 10000])
@pytest.mark.parametrize("max_points_per_voxel", [35])
@pytest.mark.parametrize("max_num_voxels", [500000])
@pytest.mark.parametrize("voxel_size", [
    (0.2, 0.2, 0.2),
    (1.4, 1.4, 1.4),
])
@pytest.mark.parametrize("coordinate_range", [
    (-5.0, -5.0, -2.0, 5.0, 5.0, 2.0),
    (-10.0, -10.0, -5.0, 10.0, 10.0, 5.0),
])
@pytest.mark.parametrize("num_features", [4, 63, 64])
@pytest.mark.parametrize("seed", [0])
def test_hard_voxelization(
    num_points: int,
    max_points_per_voxel: int,
    max_num_voxels: int,
    voxel_size: tuple[float, ...],
    coordinate_range: tuple[float, ...],
    num_features: int,
    seed: int,
) -> None:
    """Test dynamic voxelization."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    dtype = torch.float32

    # Create test points
    points = _create_test_points(num_points, coordinate_range, num_features=num_features, dtype=dtype, device=device)

    # Test PyTorch implementation
    # voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
    # voxels_pytorch = _dynamic_voxelize(
    #     points=points,
    #     voxel_size=voxel_size,
    #     coordinate_range=coordinate_range,
    #     # max_points_per_voxel=max_points_per_voxel,
    # )

    # points_cuda = points.float()

    # coors_cuda = _voxelization_mmcv(
    #     points=points_cuda,
    #     voxel_size=voxel_size,
    #     coordinate_range=coordinate_range,
    #     max_points_per_voxel=max_points_per_voxel,
    #     max_voxels=max_voxels,
    # )
    voxels_out, coors_out, num_points_per_voxel_out = _voxelization_mmcv(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_num_voxels,
    )

    # Note: for out-of-bounds points, the CUDA implementation may return empty coordinates
    # while the PyTorch implementation will return (-1, -1, -1) for those points.
    # torch.testing.assert_close(coors_cuda, voxels_pytorch, rtol=1e-5, atol=1e-5)

    # conch_num_points_per_voxel, conch_point_features = voxelization_conch(
    conch_num_points_per_voxel, conch_voxel_coords, conch_point_features = voxelization_conch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_num_voxels,
    )

    # Order isn't guaranteed
    # and if there are saturated voxels, there may be mismatches
    # for one test case: ensure no saturated voxels
    # for another: ensure that everything works when we do saturate a voxel

    # Ensure that the same number of voxels were created
    assert torch.sum(num_points_per_voxel_out) == torch.sum(conch_num_points_per_voxel)

    # Check that no voxels were saturated
    # assert torch.all(num_points_per_voxel_out < max_points_per_voxel)
    # assert torch.all(conch_num_points_per_voxel < max_points_per_voxel)

    # print(f"{conch_voxel_coords = }")
    # print(f"{coors_out = }")

    conch_coords_sorted, _ = torch.sort(conch_voxel_coords, dim=0)
    cuda_coords_sorted, _ = torch.sort(coors_out, dim=0)

    # print(f"{conch_coords_sorted = }")
    # print(f"{cuda_coords_sorted = }")

    torch.testing.assert_close(cuda_coords_sorted, conch_coords_sorted, rtol=1e-5, atol=1e-5)
    # assert False

    # if torch.all(num_points_per_voxel_out < max_points_per_voxel):
    #     assert torch.all(conch_num_points_per_voxel < max_points_per_voxel)
    #     conch_sorted, _ = torch.sort(conch_point_features, dim=0)
    #     cuda_sorted, _ = torch.sort(voxels_out, dim=0)
    #     torch.testing.assert_close(cuda_sorted, conch_sorted, rtol=1e-3, atol=1e-3)
