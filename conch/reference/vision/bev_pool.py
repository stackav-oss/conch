# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import find_spec

import torch

from conch import envs


def bev_pool_pytorch(
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


def bev_pool_backward_pytorch(
    grad_output: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    num_points, _ = geom_feats.shape
    _, _, _, _, num_channels = grad_output.shape
    num_intervals = interval_starts.size(0)

    x_grad = torch.zeros((num_points, num_channels), dtype=grad_output.dtype, device=grad_output.device)

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
        x_grad[interval_start : interval_start + interval_length] += grad_output[output_idx]

    return x_grad


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
    """Cumulative sum pooling operator for 3D voxel grids."""
    if envs.CONCH_ENABLE_CUDA_EXT:
        if find_spec("conch_cuda_ext") is None:
            raise ImportError("Conch CUDA extension is not available. Please build the extension first.")

        from conch_cuda_ext.ops.vision.bev_pool.bev_pool import bev_pool_forward as bev_pool_fwd_cuda

        return bev_pool_fwd_cuda(  # type: ignore[no-any-return]
            image_feats,
            geom_feats,
            interval_lengths,
            interval_starts,
            batch_size,
            grid_cells_z,
            grid_cells_x,
            grid_cells_y,
        )

    return bev_pool_pytorch(
        image_feats=image_feats,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
        batch_size=batch_size,
        grid_cells_z=grid_cells_z,
        grid_cells_x=grid_cells_x,
        grid_cells_y=grid_cells_y,
    )


def bev_pool_backward(
    grad_output: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
) -> torch.Tensor:
    """Cumulative sum pooling operator for 3D voxel grids."""
    if envs.CONCH_ENABLE_CUDA_EXT:
        if find_spec("conch_cuda_ext") is None:
            raise ImportError("Conch CUDA extension is not available. Please build the extension first.")

        from conch_cuda_ext.ops.vision.bev_pool.bev_pool import bev_pool_backward as bev_pool_bwd_cuda

        return bev_pool_bwd_cuda(  # type: ignore[no-any-return]
            grad_output,
            geom_feats,
            interval_lengths,
            interval_starts,
            batch_size,
            grid_cells_z,
            grid_cells_x,
            grid_cells_y,
        )

    return bev_pool_backward_pytorch(
        grad_output=grad_output,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
    )
