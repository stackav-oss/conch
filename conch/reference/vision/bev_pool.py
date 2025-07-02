# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

import torch


def bev_pool(
    image_feats: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
) -> torch.Tensor:
    _, num_channels = image_feats.shape
    num_intervals = interval_lengths.size(0)

    # B, D, H, W, C
    output = torch.zeros(
        (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels),
        dtype=image_feats.dtype,
        device=image_feats.device,
    )

    for i in range(num_intervals):
        interval_start = interval_starts[i]
        interval_length = interval_lengths[i]

        # XYZB -> BZXY
        output_idx = (
            geom_feats[interval_start, 3],
            geom_feats[interval_start, 2],
            geom_feats[interval_start, 0],
            geom_feats[interval_start, 1],
        )
        image_feats_interval_sum = image_feats[interval_start : interval_start + interval_length].sum(dim=0)

        output[output_idx] = image_feats_interval_sum

    return output
