# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Quick cumulative sum operator for 3D voxel grids."""

import torch

from conch.kernels.vision.quick_cumsum import quick_cumsum_launcher


def quick_cumsum(
    image_feats: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_lengths: torch.Tensor,
    interval_starts: torch.Tensor,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
) -> torch.Tensor:
    """Execute the quickcumsum operator.

    Args:
        image_feats: input image features, FloatTensor[n, c]
        geom_feats: input coordinates, IntTensor[n, 4]
        interval_lengths: how many points in each pooled point
        interval_starts: starting position for pooled point
        batch_size: batch size
        grid_cells_z: number of z cells of the 3D voxel grid
        grid_cells_x: number of x cells of the 3D voxel grid
        grid_cells_y: number of y cells of the 3D voxel grid.

    Returns:
        output features, FloatTensor[batch_size, grid_z_cells, grid_x_cells, grid_y_cells, c]
    """

    return quick_cumsum_launcher(
        image_feats=image_feats,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
        batch_size=batch_size,
        grid_cells_z=grid_cells_z,
        grid_cells_x=grid_cells_x,
        grid_cells_y=grid_cells_y,
    )
