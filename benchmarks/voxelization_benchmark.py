# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Voxelization benchmark."""

from typing import Final

import click
import torch

from conch.ops.vision.voxelization import VoxelizationParameter, generate_voxels
from conch.platforms import current_platform
from conch.reference.vision.voxelization import collect_point_features, voxelization_stable
from conch.utils.benchmark import BenchmarkMetadata, benchmark_it


# cuda/triton kernels parallelize on either num_points or max/actual number of voxels
# change num_points and grid_range to adjust point filtering & grid occupancy
# keep max_num_points_per_voxel less than 64 for ideal performance for this impl
# triton/cuda impl requires num_features_per_point = 4
@click.command()
@click.option(
    "--num-points",
    required=False,
    type=int,
    default=500000,
    help="Number of input points",
)
@click.option(
    "--max-num-points-per-voxel",
    required=False,
    type=int,
    default=4,
    help="Max number of points per voxel",
)
@click.option(
    "--voxel-dim",
    required=False,
    type=float,
    default=2.5,
    help="Voxel dimension same for x,y,z",
)
@click.option(
    "--grid-range",
    required=False,
    type=float,
    default=50,
    help="Grid boundary from -range to range, same for x,y,z",
)
@click.option(
    "--torch-ref",
    is_flag=True,
    help="Flag to enable Torch reference implementation for stable runs",
)
@click.option(
    "--iteration-time-ms",
    required=False,
    type=int,
    default=100,
    help="Time in milliseconds to run benchmark",
)
@click.option(
    "--warmup-time-ms",
    required=False,
    type=int,
    default=10,
    help="Time in milliseconds to warmup before recording times",
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
    "--cuda-ref",
    is_flag=True,
    help="Flag to enable CUDA reference implementation",
)
def main(
    num_points: int,
    max_num_points_per_voxel: int,
    voxel_dim: float,
    grid_range: float,
    torch_ref: bool,
    iteration_time_ms: int,
    warmup_time_ms: int,
    gpu: str,
    csv: bool,
    compile_ref: bool,
    cuda_ref: bool,
) -> None:
    """Benchmark voxelization.

    Args:
        num_points: Number of input points.
        max_num_points_per_voxel: Max number of points per voxel for output feature tensor.
        voxel_dim: Voxel dimensions for x,y,z
        grid_range: Grid boundary for x,y,z
        torch_ref: Flag to use pure torch reference implementation instead of hybrid triton/torch.
        iteration_time_ms: Time in milliseconds to run benchmark.
        warmup_time_ms: Time in milliseconds to warmup before recording times.
        gpu: Which gpu to run on.
        csv: Flag to indicate whether or not to print results in CSV format.
        compile_ref: Flag to torch.compile() the pure torch reference implementation.
        cuda_ref: Flag to enable CUDA reference implementation.
    """

    device: Final = torch.device(gpu)
    torch.set_default_device(device)

    # init points & parameters
    num_features_per_point: Final = 4
    points = torch.randn((num_points, num_features_per_point), device=device) * grid_range
    param = VoxelizationParameter(
        min_range=(-grid_range, -grid_range, -grid_range),
        max_range=(grid_range, grid_range, grid_range),
        voxel_dim=(voxel_dim, voxel_dim, voxel_dim),
        max_num_points_per_voxel=max_num_points_per_voxel,
    )

    print(f"Number of points: {num_points}")
    print(f"Grid dimensions: {param.grid_dim}")
    print(f"Max number of voxels: {param.max_num_voxels}")
    print(f"Max number of points per voxel: {param.max_num_points_per_voxel}")

    args = (points, param)

    def generate_voxels_torch(
        points: torch.Tensor, param: VoxelizationParameter
    ) -> tuple[torch.tensor, torch.tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """reference triton/torch hybrid, 2-step, stable voxelization first then generate a feature tensor."""
        use_triton = not torch_ref
        actual_num_points_per_voxel, point_raw_indices, flat_voxel_indices = voxelization_stable(
            points, param, use_triton=use_triton
        )
        point_features, capped_num_points_per_voxel = collect_point_features(
            points, actual_num_points_per_voxel, point_raw_indices, param, use_triton=use_triton
        )
        return (
            actual_num_points_per_voxel,
            point_raw_indices,
            flat_voxel_indices,
            point_features,
            capped_num_points_per_voxel,
        )

    # run base version and report voxelization stats
    actual_num_points_per_voxel, point_raw_indices, _, _, _ = generate_voxels_torch(*args)
    print("voxelization done, stats:")
    print(f"number of points within grid boundary: {point_raw_indices.shape[0]}")
    print(f"number of filled voxels: {actual_num_points_per_voxel.shape[0]}")
    print(f"Avg number of points per voxel: {torch.mean(actual_num_points_per_voxel, dtype=torch.float32)}")
    print(f"Max number of points per voxel: {torch.max(actual_num_points_per_voxel)}")
    overflow_count = (actual_num_points_per_voxel > param.max_num_points_per_voxel).int().sum()
    print(f"Number of voxels with overflowing points: {overflow_count}")

    # Benchmark implementations
    metadata = BenchmarkMetadata(
        platform=current_platform.name(),
        params={
            "num_points": num_points,
            "max_num_points_per_voxel": max_num_points_per_voxel,
            "voxel_dim": voxel_dim,
            "grid_range": grid_range,
        },
    )

    baseline_result = benchmark_it(
        lambda: generate_voxels_torch(*args),
        tag="Baseline",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    conch_result = benchmark_it(
        lambda: generate_voxels(*args),
        tag="Conch",
        metadata=metadata,
        iteration_time_ms=iteration_time_ms,
        warmup_time_ms=warmup_time_ms,
    )

    reference_cuda_result = None
    generate_voxels_cuda_fn = None
    if cuda_ref:
        from conch_cuda_ext.ops.vision.voxelization.voxelization import generate_voxels as generate_voxels_cuda

        generate_voxels_cuda_fn = generate_voxels_cuda

    if generate_voxels_cuda_fn:
        args_cuda = (
            points,
            param.min_range[0],
            param.min_range[1],
            param.min_range[2],
            param.max_range[0],
            param.max_range[1],
            param.max_range[2],
            param.voxel_dim[0],
            param.voxel_dim[1],
            param.voxel_dim[2],
            param.grid_dim[0],
            param.grid_dim[1],
            param.grid_dim[2],
            param.max_num_points_per_voxel,
            param.max_num_voxels,
        )
        reference_cuda_result = benchmark_it(
            lambda: generate_voxels_cuda_fn(*args_cuda),
            tag="CUDA",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    reference_compiled_result = None
    reference_compiled_fn = None

    if compile_ref and torch_ref:
        # Compile the reference implementation if requested
        reference_compiled_fn = torch.compile(generate_voxels_torch)

    if reference_compiled_fn:
        baseline_result = benchmark_it(
            lambda: reference_compiled_fn(*args),
            tag="Baseline (Torch compiled)",
            metadata=metadata,
            iteration_time_ms=iteration_time_ms,
            warmup_time_ms=warmup_time_ms,
        )

    conch_result.print_parameters(csv=csv)
    conch_result.print_results(csv=csv)
    baseline_result.print_results(csv=csv)
    if reference_cuda_result:
        reference_cuda_result.print_results(csv=csv)
    if reference_compiled_result:
        reference_compiled_result.print_results(csv=csv)


if __name__ == "__main__":
    main()
