# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Test BEV Pool operation."""

from typing import Final

import pytest
import torch

from conch.ops.vision.bev_pool import bev_pool as bev_pool_conch
from conch.platforms import current_platform
from conch.reference.vision.bev_pool import bev_pool as bev_pool_ref
from conch.third_party.vllm.utils import seed_everything

_BATCH_SIZES: Final = [1, 4, 8]
_NUM_POINTS: Final = [8, 100, 1000]
_NUM_CHANNELS: Final = [4, 10]
_GRID_DIMS: Final = [(16, 16, 16), (128, 128, 128)]


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
@pytest.mark.parametrize("num_points", _NUM_POINTS)
@pytest.mark.parametrize("num_channels", _NUM_CHANNELS)
@pytest.mark.parametrize("grid_dims", _GRID_DIMS)
@pytest.mark.parametrize("seed", range(3))
def test_bev_pool(
    batch_size: int, num_points: int, num_channels: int, grid_dims: tuple[int, int, int], seed: int
) -> None:
    """Test the bev_pool function."""
    device = current_platform.device
    torch.set_default_device(device)

    seed_everything(seed)

    grid_cells_z, grid_cells_x, grid_cells_y = grid_dims

    image_feats = torch.randn(num_points, num_channels, device=device, dtype=torch.float32)

    geom_feats_x = torch.randint(low=0, high=grid_cells_x, size=(num_points,), device=device, dtype=torch.long)
    geom_feats_y = torch.randint(low=0, high=grid_cells_y, size=(num_points,), device=device, dtype=torch.long)
    geom_feats_z = torch.randint(low=0, high=grid_cells_z, size=(num_points,), device=device, dtype=torch.long)
    geom_feats_b = torch.randint(low=0, high=batch_size, size=(num_points,), device=device, dtype=torch.long)
    geom_feats = torch.stack((geom_feats_x, geom_feats_y, geom_feats_z, geom_feats_b), dim=1)

    # Prepare input tensors
    ranks = (
        geom_feats[:, 0] * (grid_cells_y * grid_cells_z * batch_size)
        + geom_feats[:, 1] * (grid_cells_z * batch_size)
        + geom_feats[:, 2] * batch_size
        + geom_feats[:, 3]
    )

    indices = ranks.argsort()
    image_feats, geom_feats, ranks = image_feats[indices], geom_feats[indices], ranks[indices]
    image_feats = image_feats.contiguous()
    geom_feats = geom_feats.contiguous()

    kept = torch.ones(image_feats.shape[0], device=image_feats.device, dtype=torch.bool)
    kept[1:] = ranks[1:] != ranks[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = image_feats.shape[0] - interval_starts[-1]
    geom_feats = geom_feats.int()

    # Run the reference implementation
    output_ref = bev_pool_ref(
        image_feats=image_feats,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
        batch_size=batch_size,
        grid_cells_z=grid_cells_z,
        grid_cells_x=grid_cells_x,
        grid_cells_y=grid_cells_y,
    )

    # Run the conch implementation
    output_conch = bev_pool_conch(
        image_feats=image_feats,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
        batch_size=batch_size,
        grid_cells_z=grid_cells_z,
        grid_cells_x=grid_cells_x,
        grid_cells_y=grid_cells_y,
    )

    torch.testing.assert_close(output_ref, output_conch, atol=1e-5, rtol=1e-5)
