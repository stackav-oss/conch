"""Triton point cloud voxeliation."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class VoxelizationParameter:
    """Parameters."""

    min_range: tuple[float, float, float]
    max_range: tuple[float, float, float]
    voxel_dim: tuple[float, float, float]
    grid_dim: tuple[int, int, int]
    max_num_points_per_voxel: int
    max_num_voxels: int

    def __init__(
        self,
        min_range: tuple[float, float, float],
        max_range: tuple[float, float, float],
        voxel_dim: tuple[float, float, float],
        max_num_points_per_voxel: int,
    ) -> None:
        """Init parameters."""
        self.min_range = min_range
        self.max_range = max_range
        self.voxel_dim = voxel_dim
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.grid_dim = self._compute_grid_dim()
        self.max_num_voxels = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

    def _compute_grid_dim(self) -> tuple[int, int, int]:
        """Compute grid dimensions."""
        grid_x = round((self.max_range[0] - self.min_range[0]) / self.voxel_dim[0])
        grid_y = round((self.max_range[1] - self.min_range[1]) / self.voxel_dim[1])
        grid_z = round((self.max_range[2] - self.min_range[2]) / self.voxel_dim[2])
        return (grid_x, grid_y, grid_z)


def filter_and_label_points_torch(
    points: torch.Tensor,
    min_range: tuple[float, float, float],
    voxel_dim: tuple[float, float, float],
    grid_dim: tuple[int, int, int],
    point_voxel_indices: torch.Tensor,
) -> None:
    """Filter valid points and label each with a flat voxel index.

    Args:
        points: Input points with shape (num_points, num_features_per_point)
        min_range: Minimum bounds (min_x, min_y, min_z)
        max_range: Maximum bounds (max_x, max_y, max_z)
        voxel_dim: Voxel dimensions (voxel_dim_x, voxel_dim_y, voxel_dim_z)
        grid_dim: Grid dimensions (grid_dim_x, grid_dim_y, grid_dim_z)
        point_voxel_indices: Output flat voxel indices for each point (-1 for invalid points)
    """
    point_x = points[:, 0]
    point_y = points[:, 1]
    point_z = points[:, 2]

    min_x, min_y, min_z = min_range
    voxel_dim_x, voxel_dim_y, voxel_dim_z = voxel_dim
    grid_dim_x, grid_dim_y, grid_dim_z = grid_dim

    # Compute voxel indices
    voxel_x = torch.floor((point_x - min_x) / voxel_dim_x).to(torch.int32)
    voxel_y = torch.floor((point_y - min_y) / voxel_dim_y).to(torch.int32)
    voxel_z = torch.floor((point_z - min_z) / voxel_dim_z).to(torch.int32)

    # bounds check on voxel indices
    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = valid_x & valid_y & valid_z

    flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x

    # Store flat voxel indices for valid points
    point_voxel_indices[valid_point] = flat_voxel_idx[valid_point]


@triton.jit
def filter_and_label_points_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_points: int,
    num_features_per_point: int,
    # parameters
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    voxel_dim_x: float,
    voxel_dim_y: float,
    voxel_dim_z: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    # output
    point_voxel_indices_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Filter valid points and label each with a voxel index.

    Args:
        points_ptr: input point AoS, tensor of shape [num_points, num_features_per_point]
        voxelization parameters
        point_voxel_indices_ptr: output per point flattened voxel indices, of shape [num_points],
        tensor must be filled with invalid indices before calling this kernel.
    """
    block_idx = tl.program_id(axis=0)
    point_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    point_mask = point_idx < num_points

    point_x = tl.load(points_ptr + point_idx * num_features_per_point + 0, mask=point_mask, other=max_x + voxel_dim_x)
    point_y = tl.load(points_ptr + point_idx * num_features_per_point + 1, mask=point_mask, other=max_y + voxel_dim_y)
    point_z = tl.load(points_ptr + point_idx * num_features_per_point + 2, mask=point_mask, other=max_z + voxel_dim_z)

    voxel_x = tl.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)

    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = point_mask & valid_x & valid_y & valid_z

    flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x
    # only store valid indices and rely on otherwise default values
    tl.store(point_voxel_indices_ptr + point_idx, flat_voxel_idx, mask=valid_point)


@triton.jit
def generate_dense_voxels_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_points: int,
    # parameters
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    voxel_dim_x: float,
    voxel_dim_y: float,
    voxel_dim_z: float,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    max_num_points_per_voxel: int,
    # output
    dense_num_points_per_voxel_ptr: torch.Tensor,
    dense_point_features_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points_ptr: input points tensor, shape [num_points, num_features_per_point]
        voxelization parameters
        dense_num_points_per_voxel_ptr: output counter for number of points in each dense voxel, shape
        (max_num_voxels), tensor must be filled with 0 before calling this kernel.
        dense_point_features_ptr: output point features ordered by dense voxels, shape
        (max_num_voxels, max_num_points_per_voxel, num_features_per_point), tensor must be filled with 0 before
        calling this kernel.
    """
    block_idx = tl.program_id(axis=0)
    point_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    point_mask = point_idx < num_points

    point_x = tl.load(points_ptr + 4 * point_idx + 0, mask=point_mask, other=max_x + voxel_dim_x)
    point_y = tl.load(points_ptr + 4 * point_idx + 1, mask=point_mask, other=max_y + voxel_dim_y)
    point_z = tl.load(points_ptr + 4 * point_idx + 2, mask=point_mask, other=max_z + voxel_dim_z)
    point_w = tl.load(points_ptr + 4 * point_idx + 3, mask=point_mask, other=0)

    voxel_x = tl.floor((point_x - min_x) / voxel_dim_x).to(tl.int32)
    voxel_y = tl.floor((point_y - min_y) / voxel_dim_y).to(tl.int32)
    voxel_z = tl.floor((point_z - min_z) / voxel_dim_z).to(tl.int32)

    valid_x = (voxel_x >= 0) & (voxel_x < grid_dim_x)
    valid_y = (voxel_y >= 0) & (voxel_y < grid_dim_y)
    valid_z = (voxel_z >= 0) & (voxel_z < grid_dim_z)
    valid_point = point_mask & valid_x & valid_y & valid_z

    flat_voxel_idx = (voxel_z * grid_dim_y + voxel_y) * grid_dim_x + voxel_x
    point_idx_in_voxel = tl.atomic_add(dense_num_points_per_voxel_ptr + flat_voxel_idx, 1, mask=valid_point)

    output_idx = flat_voxel_idx * max_num_points_per_voxel + point_idx_in_voxel
    output_mask = valid_point & (point_idx_in_voxel < max_num_points_per_voxel)
    tl.store(dense_point_features_ptr + output_idx * 4 + 0, point_x, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 1, point_y, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 2, point_z, mask=output_mask)
    tl.store(dense_point_features_ptr + output_idx * 4 + 3, point_w, mask=output_mask)


@triton.jit
def generate_voxels_kernel(  # noqa: PLR0913, D417
    # input
    dense_point_features_ptr: torch.Tensor,
    dense_num_points_per_voxel_ptr: torch.Tensor,
    # parameters
    grid_dim_x: int,
    grid_dim_y: int,
    max_num_points_per_voxel: int,
    max_num_voxels: int,
    # output
    num_filled_voxels_ptr: torch.Tensor,
    num_points_per_voxel_ptr: torch.Tensor,
    point_features_ptr: torch.Tensor,
    voxel_indices_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Convert dense voxels into sparse/contiguous non-empty voxels.

    Args:
        dense_point_features_ptr: input point features tensor, shape
        (max_num_voxels, max_num_points_per_voxel, num_features_per_point)
        dense_num_points_per_voxel_ptr: input counter for number of points in each dense voxel, shape
        (max_num_voxels)
        voxelization parameters
        num_filled_voxels_ptr: output counter for the total number of non-empty voxels
        num_points_per_voxel_ptr: output counter for the number of points in each filled voxel, capped to
        max_num_points_per_voxel, shape (num_filled_voxels)
        point_features_ptr: output point features for each filled voxel, shape
        (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        voxel_indices_ptr: output per voxel 3D coordinates tensor, shape
        (num_filled_voxels, 4), only first 3 fields are set for indices in x, y, z
    """
    pid = tl.program_id(0)
    flat_voxel_idx = pid * cxpr_block_size + tl.arange(0, cxpr_block_size)

    num_points_in_voxel = tl.load(
        dense_num_points_per_voxel_ptr + flat_voxel_idx, mask=flat_voxel_idx < max_num_voxels, other=0
    )

    num_points_in_voxel = tl.minimum(num_points_in_voxel, max_num_points_per_voxel)
    valid_voxel = num_points_in_voxel > 0

    voxel_idx = tl.atomic_add(num_filled_voxels_ptr + tl.zeros_like(valid_voxel), 1, mask=valid_voxel)

    # store num_points_per_voxel with clipping
    tl.store(num_points_per_voxel_ptr + voxel_idx, num_points_in_voxel, mask=valid_voxel)

    # convert flat voxel index to 3d coordinates
    voxel_x = flat_voxel_idx % grid_dim_x
    voxel_y = (flat_voxel_idx // grid_dim_x) % grid_dim_y
    voxel_z = flat_voxel_idx // (grid_dim_y * grid_dim_x)
    # store 3d indices
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 0, voxel_x, mask=valid_voxel)
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 1, voxel_y, mask=valid_voxel)
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 2, voxel_z, mask=valid_voxel)
    # copy from nvidia opensource code, index is padded to int4 type
    tl.store(voxel_indices_ptr + voxel_idx * 4 + 3, 0, mask=valid_voxel)

    # store all feature points, even if they are 0 because Triton
    for point_idx in range(0, max_num_points_per_voxel, 1):
        input_idx = flat_voxel_idx * max_num_points_per_voxel + point_idx
        point_x = tl.load(dense_point_features_ptr + input_idx * 4 + 0, mask=valid_voxel)
        point_y = tl.load(dense_point_features_ptr + input_idx * 4 + 1, mask=valid_voxel)
        point_z = tl.load(dense_point_features_ptr + input_idx * 4 + 2, mask=valid_voxel)
        point_w = tl.load(dense_point_features_ptr + input_idx * 4 + 3, mask=valid_voxel)

        output_idx = voxel_idx * max_num_points_per_voxel + point_idx
        tl.store(point_features_ptr + output_idx * 4 + 0, point_x, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 1, point_y, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 2, point_z, mask=valid_voxel)
        tl.store(point_features_ptr + output_idx * 4 + 3, point_w, mask=valid_voxel)


def collect_point_features_torch(  # noqa: PLR0913, D417
    points: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    segment_offsets: torch.Tensor,
    point_indices: torch.Tensor,
    max_num_points_per_voxel: int,
    point_features: torch.Tensor,
    capped_num_points_per_voxel: torch.Tensor,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points: input points tensor, shape (num_points, num_features_per_point)
        num_points_per_voxel: input number of points per voxel tensor, shape (num_filled_voxels)
        segment_offsets: input segment end offsets, shape (num_filled_voxels)
        point_indices: input raw point indices, shape (num_valid_points)
        voxelization parameters
        point_features: output voxel point features, shape (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        capped_num_points_per_voxel: output number of points per voxel tensor after capping, shape (num_filled_voxels)
    """
    # inclusive sum to exclusive sum
    start_indices = torch.cat(
        (torch.zeros(1, dtype=segment_offsets.dtype, device=segment_offsets.device), segment_offsets[:-1])
    )
    capped_num_points_per_voxel[:] = torch.where(
        num_points_per_voxel > max_num_points_per_voxel, max_num_points_per_voxel, num_points_per_voxel
    )
    # init feature tensor with 0 first
    point_features.zero_()

    # top n filtering
    for voxel_idx, (start_idx, num_points_in_voxel) in enumerate(
        zip(start_indices, capped_num_points_per_voxel, strict=False)
    ):
        raw_indices = point_indices[start_idx : start_idx + num_points_in_voxel]
        for point_idx_in_voxel, raw_point_idx in enumerate(raw_indices):
            point_features[voxel_idx, point_idx_in_voxel, :] = points[raw_point_idx, :]


@triton.jit
def collect_point_features_kernel(  # noqa: PLR0913, D417
    # input
    points_ptr: torch.Tensor,
    num_features_per_point: int,
    segment_offsets_ptr: torch.Tensor,
    num_filled_voxels: int,
    point_indices_ptr: torch.Tensor,
    # parameters
    max_num_points_per_voxel: int,
    # output
    point_features_ptr: torch.Tensor,
    capped_num_points_per_voxel_ptr: torch.Tensor,
    # Constants
    cxpr_block_size: tl.constexpr,
) -> None:
    """Group valid points into dense voxels.

    Args:
        points_ptr: input points tensor, shape (num_points, num_features_per_point)
        segment_offsets_ptr: input segment end offsets, shape (num_filled_voxels)
        point_indices_ptr: input raw point indices, shape (num_valid_points)
        voxelization parameters
        point_features_ptr: output voxel point features, shape (num_filled_voxels, max_num_points_per_voxel, num_features_per_point)
        capped_num_points_per_voxel_ptr: output number of points per voxel tensor after capping, shape (num_filled_voxels)
    """
    block_idx = tl.program_id(axis=0)
    voxel_idx = block_idx * cxpr_block_size + tl.arange(0, cxpr_block_size)
    voxel_mask = voxel_idx < num_filled_voxels

    # top n filtering
    segment_start = tl.load(segment_offsets_ptr + voxel_idx - 1, mask=(voxel_mask & (voxel_idx > 0)), other=0)
    segment_end = tl.load(segment_offsets_ptr + voxel_idx, mask=voxel_mask, other=0)
    num_points_in_voxel = segment_end - segment_start
    num_points_in_voxel = tl.minimum(num_points_in_voxel, max_num_points_per_voxel)
    tl.store(capped_num_points_per_voxel_ptr + voxel_idx, num_points_in_voxel, mask=voxel_mask)

    for voxel_point_idx in range(0, max_num_points_per_voxel, 1):
        # this mask is sufficient since other num_points_in_voxel == 0
        per_voxel_mask = voxel_point_idx < num_points_in_voxel

        raw_point_idx = tl.load(point_indices_ptr + segment_start + voxel_point_idx, mask=per_voxel_mask)
        output_idx = voxel_idx * max_num_points_per_voxel + voxel_point_idx
        # tl.make_block_ptr()?
        for feature_idx in range(0, num_features_per_point, 1):
            value = tl.load(
                points_ptr + raw_point_idx * num_features_per_point + feature_idx, mask=per_voxel_mask, other=0
            )
            tl.store(point_features_ptr + output_idx * num_features_per_point + feature_idx, value, mask=voxel_mask)


def generate_voxels_triton(
    points: torch.Tensor, param: VoxelizationParameter
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates voxels from input points, output voxels and points are randomly ordered due to use of atomics.

    Args:
        points: input points; expected dimensions (num_points, 4), last dimension should have fields of x,y,z,_.
        param: parameters.

    Returns:
        tuple of voxels SoA:
            capped_num_points_per_voxel, shape [num_filled_voxels], note per voxel point counters are capped with max_num_points_per_voxel.
            point_features, shape [num_filled_voxels, max_num_points_per_voxel, num_features_per_point],
            empty points are filled with 0.
            voxel_indices, shape [num_filled_voxels, 4], only first 3 fields are used for x,y,z indices.
    """
    assert points.is_cuda
    device = points.device
    num_points, num_features_per_point = points.shape
    assert num_features_per_point == 4  # noqa: PLR2004
    # same as original nvidia cuda impl
    num_elements_per_voxel_index = 4

    # dense (must set to 0s)
    dense_num_points_per_voxel = torch.zeros((param.max_num_voxels), dtype=torch.int32, device=device)
    dense_point_features = torch.zeros(
        (param.max_num_voxels, param.max_num_points_per_voxel, num_features_per_point), dtype=torch.float, device=device
    )

    # sparse/contiguous output
    num_filled_voxels = torch.zeros((1), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.empty_like(dense_num_points_per_voxel)
    point_features = torch.empty_like(dense_point_features)
    voxel_indices = torch.empty((param.max_num_voxels, num_elements_per_voxel_index), dtype=torch.int32, device=device)

    block_size = 256
    num_threads_per_warp = 32

    # first generate dense voxels
    grid = (triton.cdiv(num_points, block_size),)
    generate_dense_voxels_kernel[grid](
        points,
        num_points,
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
        dense_num_points_per_voxel,
        dense_point_features,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
    )

    # compress into contiguous/sparse filled voxels
    grid = (triton.cdiv(param.max_num_voxels, block_size),)
    generate_voxels_kernel[grid](
        dense_point_features,
        dense_num_points_per_voxel,
        param.grid_dim[0],
        param.grid_dim[1],
        param.max_num_points_per_voxel,
        param.max_num_voxels,
        num_filled_voxels,
        num_points_per_voxel,
        point_features,
        voxel_indices,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
    )

    total_filled_voxels = num_filled_voxels.cpu()[0]
    print(f"num filled voxels {total_filled_voxels}")

    return (
        num_points_per_voxel[:total_filled_voxels],
        point_features[:total_filled_voxels, :, :],
        voxel_indices[:total_filled_voxels, :],
    )


def voxelization_stable(
    points: torch.Tensor, param: VoxelizationParameter, use_triton: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize input points. output is deterministic running back to back with the same input.

    Args:
        points: input points; expected dimensions (num_points, num_features_per_point), first three fields of last
        dimentions should be x, y, z.
        param: voxelization parameters.
        use_triton: whether to use a Triton kernel for labeling points

    Returns:
        tuple, voxels SoA sorted by flat voxel indices on the grid:
            num_points_per_voxel, shape [num_filled_voxels], note this is actual number of points in each voxel without clipping.
            point_indices: original point indices grouped by voxels, shape [num_valid_points], points within the same voxel are
            contiguous with segment size specified in num_points_per_voxel.
            flat_voxel_indices, shape [num_filled_voxels].
    """
    assert points.is_cuda
    device = points.device
    num_points, num_features_per_point = points.shape

    # init raw indices
    point_raw_indices = torch.arange(num_points).to(device)
    # set default output voxel indices to invalid max
    point_voxel_indices = torch.full((num_points,), param.max_num_voxels, dtype=torch.int32, device=device)

    if use_triton:
        # compute point wise flat voxel indices
        block_size = 256
        num_threads_per_warp = 32
        grid = (triton.cdiv(num_points, block_size),)
        filter_and_label_points_kernel[grid](
            points,
            num_points,
            num_features_per_point,
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
            point_voxel_indices,
            cxpr_block_size=block_size,
            num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
        )
    else:
        filter_and_label_points_torch(points, param.min_range, param.voxel_dim, param.grid_dim, point_voxel_indices)

    # mask for points within bound, can skip select step when most points are valid
    mask = point_voxel_indices < param.max_num_voxels
    # value
    raw_indices_selected = torch.masked_select(point_raw_indices, mask)
    # key
    voxel_indices_selected = torch.masked_select(point_voxel_indices, mask)
    print(f"num valid points {raw_indices_selected.size(dim=0)} total {num_points}")

    # group points into voxels with sort_by_key(), use stable to keep original points ordering
    sorted_voxel_indices, permute_indices = torch.sort(voxel_indices_selected, stable=True)
    sorted_raw_indices = raw_indices_selected[permute_indices]
    # run length encode
    voxel_indices, num_points_per_voxel = torch.unique_consecutive(sorted_voxel_indices, return_counts=True)

    print(f"num filled voxels {voxel_indices.size(dim=0)}")
    return num_points_per_voxel.to(torch.int32), sorted_raw_indices, voxel_indices


def collect_point_features(
    points: torch.Tensor,
    num_points_per_voxel: torch.Tensor,
    point_indices: torch.Tensor,
    param: VoxelizationParameter,
    use_triton: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather voxel point features from original points and voxelization result.

    Args:
        points: input points, expected dimensions (num_points, num_features_per_point).
        num_points_per_voxel: shape [num_filled_voxels], actual number of points in each voxel.
        point_indices: original point indices grouped by voxels, shape [num_valid_points]
        param: voxelization parameters.
        use_triton: whether to use a Triton kernel

    Returns:
        point_features: shape [num_valid_points, max_num_points_per_voxel, num_features_per_point], empty points are
        filled with 0.
        capped_num_points_per_voxel: shape [num_filled_voxels], number of points in each voxel after max capping.
    """
    assert points.is_cuda
    device = points.device
    num_points, num_features_per_point = points.shape
    assert num_features_per_point == 4  # noqa: PLR2004

    (num_filled_voxels,) = num_points_per_voxel.shape
    assert num_filled_voxels <= param.max_num_voxels
    (num_valid_points,) = point_indices.shape
    assert num_valid_points <= num_points

    segment_offsets = torch.cumsum(num_points_per_voxel, dim=0)

    # output
    capped_num_points_per_voxel = torch.empty_like(num_points_per_voxel)
    point_features = torch.empty(
        (num_filled_voxels, param.max_num_points_per_voxel, num_features_per_point), dtype=torch.float, device=device
    )

    if use_triton:
        # one thread per voxel, when max_num_points_per_voxel is larger than 64, use one block per voxel
        block_size = 256
        num_threads_per_warp = 32
        grid = (triton.cdiv(num_filled_voxels, block_size),)
        collect_point_features_kernel[grid](
            points,
            num_features_per_point,
            segment_offsets,
            num_filled_voxels,
            point_indices,
            param.max_num_points_per_voxel,
            point_features,
            capped_num_points_per_voxel,
            cxpr_block_size=block_size,
            num_warps=block_size // num_threads_per_warp,  # pyright: ignore[reportCallIssue]
        )
    else:
        collect_point_features_torch(
            points,
            num_points_per_voxel,
            segment_offsets,
            point_indices,
            param.max_num_points_per_voxel,
            point_features,
            capped_num_points_per_voxel,
        )

    return point_features, capped_num_points_per_voxel
