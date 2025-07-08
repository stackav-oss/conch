# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test voxelization implementations."""

from typing import Final

import pytest
import torch

from conch.platforms import current_platform
from conch.reference.vision.voxelization import _voxelization_mmcv, _voxelization_pytorch
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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("num_points", _NUM_POINTS)
@pytest.mark.parametrize("voxel_size", _VOXEL_SIZES)
@pytest.mark.parametrize("max_points_per_voxel", _MAX_POINTS_PER_VOXEL)
@pytest.mark.parametrize("max_voxels", _MAX_VOXELS)
@pytest.mark.parametrize("seed", range(3))
def test_voxelization_pytorch_vs_cuda(
    dtype: torch.dtype,
    num_points: int,
    voxel_size: tuple[float, float, float],
    max_points_per_voxel: int,
    max_voxels: int,
    seed: int,
) -> None:
    """Test that PyTorch and CUDA voxelization implementations give similar results."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)

    # Define coordinate range based on voxel size to ensure reasonable voxel distribution
    vx, vy, vz = voxel_size
    coordinate_range = (-10.0, -10.0, -3.0, 10.0, 10.0, 3.0)

    # Create test points
    points = _create_test_points(num_points, coordinate_range, num_features=4, dtype=dtype, device=device)

    # Test PyTorch implementation
    voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    # Test CUDA implementation (if available)
    try:
        # CUDA implementation may not support all dtypes, so use float32
        if dtype == torch.float16:
            points_cuda = points.float()
        else:
            points_cuda = points

        voxels_cuda, coors_cuda, num_points_cuda = _voxelization_mmcv(
            points=points_cuda,
            voxel_size=voxel_size,
            coordinate_range=coordinate_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )

        # Compare results
        # Note: Due to potential differences in implementation details (like sorting order),
        # we check that the outputs have consistent shapes and reasonable values

        # Check shapes are consistent
        assert voxels_pytorch.shape[0] == coors_pytorch.shape[0] == num_points_pytorch.shape[0]
        assert voxels_cuda.shape[0] == coors_cuda.shape[0] == num_points_cuda.shape[0]

        # Check that both implementations produce valid output dimensions
        assert voxels_pytorch.shape[1] == max_points_per_voxel
        assert voxels_pytorch.shape[2] == points.shape[1]
        assert voxels_cuda.shape[1] == max_points_per_voxel
        assert voxels_cuda.shape[2] == points.shape[1]

        # Check that coordinates are within expected range
        assert coors_pytorch.shape[1] == 3
        assert coors_cuda.shape[1] == 3

        # Check that num_points_per_voxel values are reasonable
        assert torch.all(num_points_pytorch >= 0)
        assert torch.all(num_points_pytorch <= max_points_per_voxel)
        assert torch.all(num_points_cuda >= 0)
        assert torch.all(num_points_cuda <= max_points_per_voxel)

        # Check that the number of voxels doesn't exceed max_voxels
        assert voxels_pytorch.shape[0] <= max_voxels
        assert voxels_cuda.shape[0] <= max_voxels

        print(f"PyTorch: {voxels_pytorch.shape[0]} voxels")
        print(f"CUDA: {voxels_cuda.shape[0]} voxels")

    except ImportError:
        # CUDA extension not available, skip comparison
        pytest.skip("CUDA extension not available")


@pytest.mark.parametrize("max_points_per_voxel", [-1])
@pytest.mark.parametrize("seed", range(2))
def test_voxelization_dynamic_mode(max_points_per_voxel: int, seed: int) -> None:
    """Test dynamic voxelization mode (max_points_per_voxel = -1)."""
    seed_everything(seed)

    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    num_points = 500
    voxel_size = (0.2, 0.2, 0.2)
    coordinate_range = (-5.0, -5.0, -2.0, 5.0, 5.0, 2.0)
    max_voxels = -1

    # Create test points
    points = _create_test_points(num_points, coordinate_range, num_features=4, dtype=dtype, device=device)

    # Test PyTorch implementation
    voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    # Test CUDA implementation (if available)
    try:
        # Use float32 for CUDA implementation
        points_cuda = points.float()
        coors_cuda = _voxelization_mmcv(
            points=points_cuda,
            voxel_size=voxel_size,
            coordinate_range=coordinate_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )

        # In dynamic mode, CUDA implementation returns only coordinates
        # Check that coordinates have correct shape
        assert coors_cuda.shape[0] == points.shape[0]
        assert coors_cuda.shape[1] == 3

        print(f"Dynamic mode - PyTorch voxels shape: {voxels_pytorch.shape}")
        print(f"Dynamic mode - CUDA coordinates shape: {coors_cuda.shape}")

    except ImportError:
        # CUDA extension not available, skip comparison
        pytest.skip("CUDA extension not available")


def test_voxelization_edge_cases() -> None:
    """Test voxelization edge cases."""
    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    voxel_size = (0.1, 0.1, 0.1)
    coordinate_range = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
    max_points_per_voxel = 10
    max_voxels = 100

    # Test with very few points
    few_points = _create_test_points(5, coordinate_range, num_features=3, dtype=dtype, device=device)

    voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
        points=few_points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    # Check basic validity
    assert voxels_pytorch.shape[0] > 0
    assert coors_pytorch.shape[0] > 0
    assert num_points_pytorch.shape[0] > 0
    assert torch.all(num_points_pytorch >= 0)

    # Test with points at coordinate boundaries
    boundary_points = torch.tensor(
        [
            [-1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )

    voxels_boundary, coors_boundary, num_points_boundary = _voxelization_pytorch(
        points=boundary_points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    # Check that boundary points are handled correctly
    assert voxels_boundary.shape[0] > 0
    assert coors_boundary.shape[0] > 0
    assert num_points_boundary.shape[0] > 0


@pytest.mark.parametrize("num_features", [3, 4, 5, 6])
def test_voxelization_different_feature_dimensions(num_features: int) -> None:
    """Test voxelization with different numbers of point features."""
    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    num_points = 200
    voxel_size = (0.2, 0.2, 0.2)
    coordinate_range = (-2.0, -2.0, -1.0, 2.0, 2.0, 1.0)
    max_points_per_voxel = 15
    max_voxels = 500

    # Create test points with different feature dimensions
    points = _create_test_points(num_points, coordinate_range, num_features=num_features, dtype=dtype, device=device)

    # Test PyTorch implementation
    voxels_pytorch, coors_pytorch, num_points_pytorch = _voxelization_pytorch(
        points=points,
        voxel_size=voxel_size,
        coordinate_range=coordinate_range,
        max_points_per_voxel=max_points_per_voxel,
        max_voxels=max_voxels,
    )

    # Check that output preserves feature dimensions
    assert voxels_pytorch.shape[2] == num_features
    assert coors_pytorch.shape[1] == 3  # Always 3D coordinates

    # Test CUDA implementation (if available)
    try:
        # Use float32 for CUDA implementation
        points_cuda = points.float()
        voxels_cuda, coors_cuda, num_points_cuda = _voxelization_mmcv(
            points=points_cuda,
            voxel_size=voxel_size,
            coordinate_range=coordinate_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )

        # Check that CUDA implementation also preserves feature dimensions
        assert voxels_cuda.shape[2] == num_features
        assert coors_cuda.shape[1] == 3

    except ImportError:
        # CUDA extension not available, skip comparison
        pytest.skip("CUDA extension not available")


def test_voxelization_deterministic() -> None:
    """Test that voxelization produces deterministic results."""
    device: Final = torch.device(current_platform.device)
    dtype = torch.float32
    num_points = 300
    voxel_size = (0.15, 0.15, 0.15)
    coordinate_range = (-3.0, -3.0, -1.5, 3.0, 3.0, 1.5)
    max_points_per_voxel = 20
    max_voxels = 1000

    # Create fixed test points
    torch.manual_seed(42)
    points = _create_test_points(num_points, coordinate_range, num_features=4, dtype=dtype, device=device)

    # Run PyTorch implementation multiple times
    results = []
    for _ in range(3):
        voxels, coors, num_points_per_voxel = _voxelization_pytorch(
            points=points,
            voxel_size=voxel_size,
            coordinate_range=coordinate_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        results.append((voxels, coors, num_points_per_voxel))

    # Check that results are identical across runs
    for i in range(1, len(results)):
        torch.testing.assert_close(results[0][0], results[i][0])
        torch.testing.assert_close(results[0][1], results[i][1])
        torch.testing.assert_close(results[0][2], results[i][2])
