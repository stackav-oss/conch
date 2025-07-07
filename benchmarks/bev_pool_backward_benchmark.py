# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""BEV Pool backward pass benchmark."""

import sys
from typing import Final

import click
import torch

from conch.ops.vision.bev_pool import bev_pool_backward as bev_pool_backward_conch
from conch.platforms import current_platform
from conch.reference.vision.bev_pool import bev_pool_backward as bev_pool_backward_ref
from conch.third_party.vllm.utils import seed_everything
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


def _create_bev_pool_backward_data(
    num_points: int,
    num_channels: int,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test data for BEV Pool backward operation.

    Args:
        num_points: Number of input points.
        num_channels: Number of feature channels per point.
        batch_size: Number of batches.
        grid_cells_z: Number of Z grid cells.
        grid_cells_x: Number of X grid cells.
        grid_cells_y: Number of Y grid cells.
        device: Device to create tensors on.

    Returns:
        Tuple of (grad_output, geom_feats, interval_starts, interval_lengths).
    """
    # Create random gradient output in the shape of BEV pool output
    grad_output = torch.randn(
        batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels, device=device, dtype=torch.float32
    )

    # Create geometry features with random but valid coordinates
    geom_feats = torch.stack(
        [
            torch.randint(0, grid_cells_x, (num_points,), device=device),  # X coordinate
            torch.randint(0, grid_cells_y, (num_points,), device=device),  # Y coordinate
            torch.randint(0, grid_cells_z, (num_points,), device=device),  # Z coordinate
            torch.randint(0, batch_size, (num_points,), device=device),  # Batch index
        ],
        dim=1,
    ).long()

    # Create a linear index for sorting and grouping
    linear_indices = (
        geom_feats[:, 3] * (grid_cells_z * grid_cells_x * grid_cells_y)  # batch
        + geom_feats[:, 2] * (grid_cells_x * grid_cells_y)  # z
        + geom_feats[:, 1] * grid_cells_y  # x
        + geom_feats[:, 0]  # y
    )

    # Sort by linear indices to group points in same voxels
    sorted_indices = torch.argsort(linear_indices)
    sorted_linear_indices = linear_indices[sorted_indices]

    # Find unique voxels and create intervals
    unique_indices, counts = torch.unique_consecutive(sorted_linear_indices, return_counts=True)
    num_intervals = len(unique_indices)

    # Create interval starts and lengths
    interval_starts = torch.zeros(num_intervals, device=device, dtype=torch.long)
    interval_lengths = counts.long()

    current_start = 0
    for i in range(num_intervals):
        interval_starts[i] = current_start
        current_start += interval_lengths[i]

    # Reorder geometry by sorted indices
    geom_feats = geom_feats[sorted_indices]

    return grad_output, geom_feats, interval_starts, interval_lengths


@click.command()
@click.option(
    "--num-points",
    required=False,
    type=int,
    default=10000,
    help="Number of input points",
)
@click.option(
    "--num-channels",
    required=False,
    type=int,
    default=64,
    help="Number of feature channels per point",
)
@click.option(
    "--batch-size",
    required=False,
    type=int,
    default=2,
    help="Batch size",
)
@click.option(
    "--grid-cells-z",
    required=False,
    type=int,
    default=16,
    help="Number of Z grid cells",
)
@click.option(
    "--grid-cells-x",
    required=False,
    type=int,
    default=200,
    help="Number of X grid cells",
)
@click.option(
    "--grid-cells-y",
    required=False,
    type=int,
    default=200,
    help="Number of Y grid cells",
)
@click.option(
    "--iteration-time-ms",
    required=False,
    type=int,
    default=10000,
    help="Time in milliseconds to run benchmark",
)
@click.option(
    "--warmup-time-ms",
    required=False,
    type=int,
    default=1000,
    help="Time in milliseconds to warmup before recording times",
)
@click.option(
    "--absolute-tolerance",
    required=False,
    type=float,
    default=1e-3,
    help="Absolute tolerance to match with",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Flag for printing verbose output",
)
@click.option(
    "--gpu",
    required=False,
    type=str,
    default=current_platform.device,
    help="Device to run on",
)
@click.option(
    "--csv",
    is_flag=True,
    help="Flag for printing results in CSV format",
)
@click.option(
    "--compile-ref",
    is_flag=True,
    help="Flag to torch.compile() the reference impl",
)
@click.option(
    "--compile-conch",
    is_flag=True,
    help="Flag to torch.compile() the Conch impl",
)
def main(
    num_points: int,
    num_channels: int,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
    iteration_time_ms: int,
    warmup_time_ms: int,
    absolute_tolerance: float,
    verbose: bool,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    compile_conch: bool,
) -> None:
    """Benchmark BEV Pool backward pass.

    Args:
        num_points: Number of input points.
        num_channels: Number of feature channels per point.
        batch_size: Batch size.
        grid_cells_z: Number of Z grid cells.
        grid_cells_x: Number of X grid cells.
        grid_cells_y: Number of Y grid cells.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        absolute_tolerance: Absolute tolerance used to check accuracy.
        verbose: Flag to indicate whether or not to print verbose output.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
        compile_ref: Flag to torch.compile() the reference implementation.
        compile_conch: Flag to torch.compile() the Conch implementation.
    """
    seed: Final = 0
    seed_everything(seed)

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "num_points": num_points,
            "num_channels": num_channels,
            "batch_size": batch_size,
            "grid_cells_z": grid_cells_z,
            "grid_cells_x": grid_cells_x,
            "grid_cells_y": grid_cells_y,
        },
    )

    # Create test data
    grad_output, geom_feats, interval_starts, interval_lengths = _create_bev_pool_backward_data(
        num_points=num_points,
        num_channels=num_channels,
        batch_size=batch_size,
        grid_cells_z=grid_cells_z,
        grid_cells_x=grid_cells_x,
        grid_cells_y=grid_cells_y,
        device=device,
    )

    # Compile functions if requested
    bev_pool_backward_ref_fn = torch.compile(bev_pool_backward_ref) if compile_ref else bev_pool_backward_ref
    bev_pool_backward_conch_fn = torch.compile(bev_pool_backward_conch) if compile_conch else bev_pool_backward_conch

    # Test both implementations
    args = (
        grad_output,
        geom_feats,
        interval_starts,
        interval_lengths,
    )

    ref_output = bev_pool_backward_ref_fn(*args)
    conch_output = bev_pool_backward_conch_fn(*args)

    # Accuracy checks
    if not torch.allclose(ref_output, conch_output, atol=absolute_tolerance):
        print(f"WARNING: Reference and Conch results differ! (atol={absolute_tolerance})", file=sys.stderr)
        print(f"Output max diff: {(ref_output - conch_output).abs().max().item()}", file=sys.stderr)
        print(f"Ref shape: {ref_output.shape}, Conch shape: {conch_output.shape}", file=sys.stderr)

        if verbose:
            print(f"Reference output: {ref_output}", file=sys.stderr)
            print(f"Conch output: {conch_output}", file=sys.stderr)
    else:
        print(f"Reference vs Conch: Results matched with atol={absolute_tolerance} :)", file=sys.stderr)

    # Benchmark implementations
    baseline_result = benchmark_it(
        lambda: bev_pool_backward_ref_fn(*args),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: bev_pool_backward_conch_fn(*args),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    # Print results
    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
