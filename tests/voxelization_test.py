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
    num_features: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
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


@pytest.mark.parametrize("max_points_per_voxel", [35])
# @pytest.mark.parametrize("max_num_voxels", [20000])
# @pytest.mark.parametrize("num_points", [64])
# @pytest.mark.parametrize("num_points", [3])
# @pytest.mark.parametrize("num_points", [10])
# @pytest.mark.parametrize("num_points", [100])
@pytest.mark.parametrize("num_points", [1000, 20000])
# @pytest.mark.parametrize("max_points_per_voxel", [3])
# @pytest.mark.parametrize("max_num_voxels", [10])
# @pytest.mark.parametrize("max_num_voxels", [10])
# @pytest.mark.parametrize("max_num_voxels", [100000])
@pytest.mark.parametrize("max_num_voxels", [50000])
@pytest.mark.parametrize("num_features", [4, 63, 64])
# @pytest.mark.parametrize("num_features", [4])
@pytest.mark.parametrize("seed", range(2))
# @pytest.mark.parametrize("seed", [0])
# @pytest.mark.parametrize("num_features", [3, 4, 63, 64])
def test_hard_voxelization(
    num_points: int,
    max_points_per_voxel: int,
    max_num_voxels: int,
    num_features: int,
    seed: int,
) -> None:
    """Test dynamic voxelization."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    # num_points = 500
    # x, y, z
    voxel_size = (0.2, 0.2, 0.2)
    # x_min, y_min, z_min, x_max, y_max, z_max
    coordinate_range = (-5.0, -5.0, -2.0, 5.0, 5.0, 2.0)
    # max_voxels = -1
    # _ = max_points_per_voxel

    # Create test points
    points = _create_test_points(num_points, coordinate_range, num_features=num_features, dtype=dtype, device=device)
    # points = torch.ones_like(points)
    # points = torch.tensor([
    #     [-5.0, -5.0, -2.0, 69.0],
    #     [-5.0, -5.0, -2.0, 69.0],
    # ], dtype=dtype, device=device)

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
        # points=points_cuda,
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_num_voxels,
    )

    # print(f"{voxels_out = }")
    # print(f"{coors_out = }")
    # print(f"{num_points_per_voxel_out = }")

    # Note: for out-of-bounds points, the CUDA implementation may return empty coordinates
    # while the PyTorch implementation will return (-1, -1, -1) for those points.
    # torch.testing.assert_close(coors_cuda, voxels_pytorch, rtol=1e-5, atol=1e-5)

    conch_num_points_per_voxel, conch_point_features = voxelization_conch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_num_voxels,
    )

    # print(f"{conch_num_points_per_voxel = }")

    # print(f"{voxels_out.shape = }")
    # print(f"{conch_point_features.shape = }")

    # print(f"{voxels_out = }")
    # print(f"{conch_point_features = }")

    # print(f"{num_points_per_voxel_out = }")
    # print(f"{conch_num_points_per_voxel = }")

    # nonzero = torch.squeeze(torch.nonzero(conch_num_points_per_voxel))
    # print(f"{torch.nonzero(conch_num_points_per_voxel) = }")
    # print(f"{conch_num_points_per_voxel[nonzero] = }")

    # print(f"{conch_point_features[nonzero] = }")

    # num_voxels = len(nonzero)

    # conch_result, _ = torch.sort(conch_point_features[nonzero], dim=0)
    # cuda_result, _ = torch.sort(voxels_out[:num_voxels], dim=0)

    # print(f"{conch_result = }")
    # print(f"{cuda_result = }")

    # print(f"{voxels_out[:num_voxels] = }")
    # print(f"{voxels_out[:num_voxels].shape = }")
    # print(f"{torch.sort(voxels_out[:num_voxels]) = }")
    # print(f"{conch_point_features.shape = }")
    # print(f"{nonzero.shape = }")
    # print(f"{conch_point_features[nonzero].shape = }")

    # assert False

    print(f"{num_points_per_voxel_out = }")
    print(f"{voxels_out = }")

    print(f"{conch_num_points_per_voxel = }")
    print(f"{conch_point_features = }")

    # torch.testing.assert_close(voxels_out, conch_point_features, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(cuda_result, conch_result, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(num_points_per_voxel_out, conch_num_points_per_voxel, rtol=1e-5, atol=1e-5)
    # torch.testing.assert_close(voxels_out, conch_point_features, rtol=1e-5, atol=1e-5)
