// Copyright 2025 Stack AV Co.
// SPDX-License-Identifier: Apache-2.0

// derived from https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/src/pointpillar/lidar-voxelization.cu

__global__ void generate_dense_voxels_kernel(
    const float4 *points,
    int num_points,
    float min_x_range,
    float min_y_range,
    float min_z_range,
    float max_x_range,
    float max_y_range,
    float max_z_range,
    float voxel_x_size,
    float voxel_y_size,
    float voxel_z_size,
    int grid_x_size,
    int grid_y_size,
    int grid_z_size,
    int max_num_points_per_voxel,
    int *num_points_per_dense_voxel,
    float4 *dense_voxels)
{
  const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= num_points)
    return;

  const auto point = points[point_idx];

  if(point.x<min_x_range||point.x>=max_x_range
    || point.y<min_y_range||point.y>=max_y_range
    || point.z<min_z_range||point.z>=max_z_range)
    return;

  const int voxel_x = floorf((point.x - min_x_range)/voxel_x_size);
  const int voxel_y = floorf((point.y - min_y_range)/voxel_y_size);
  const int voxel_z = floorf((point.z - min_z_range)/voxel_z_size);
  const auto voxel_idx = (voxel_z * grid_y_size + voxel_y) * grid_x_size + voxel_x;

  const auto point_idx_in_voxel = atomicAdd(num_points_per_dense_voxel + voxel_idx, 1);

  if(point_idx_in_voxel < max_num_points_per_voxel)
    dense_voxels[voxel_idx*max_num_points_per_voxel + point_idx_in_voxel] = point;
}

void generate_dense_voxels(
    const float *points,
    int num_points,
    float min_x_range,
    float min_y_range,
    float min_z_range,
    float max_x_range,
    float max_y_range,
    float max_z_range,
    float voxel_x_size,
    float voxel_y_size,
    float voxel_z_size,
    int grid_x_size,
    int grid_y_size,
    int grid_z_size,
    int max_num_points_per_voxel,
    int *num_points_per_dense_voxel,
    float *dense_voxels)
{
  constexpr int block_dim = 256;
  const int grid_dim = (num_points+block_dim-1)/block_dim;
  generate_dense_voxels_kernel<<<grid_dim, block_dim, 0, nullptr>>>
    (reinterpret_cast<const float4*>(points),
     num_points,
     min_x_range,
     min_y_range,
     min_z_range,
     max_x_range,
     max_y_range,
     max_z_range,
     voxel_x_size,
     voxel_y_size,
     voxel_z_size,
     grid_x_size,
     grid_y_size,
     grid_z_size,
     max_num_points_per_voxel,
     num_points_per_dense_voxel,
     reinterpret_cast<float4*>(dense_voxels));
}

__global__ void generate_base_features_kernel(
    const int* num_points_per_dense_voxel,
    const float4* dense_voxels,
    int grid_x_size,
    int grid_y_size,
    int grid_z_size,
    int max_num_voxels,
    int max_num_points_per_voxel,
    int* num_filled_voxels,
    float4* voxel_features,
    int4* voxel_indices,
    int* num_points_per_voxel)
{
  const auto dense_voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(dense_voxel_idx >= max_num_voxels)
    return;

//  const auto dense_voxel_idx = voxel_y * grid_x_size + voxel_x;

  auto num_points_in_voxel = num_points_per_dense_voxel[dense_voxel_idx];
  if(num_points_in_voxel==0)
    return;
  //num_points_in_voxel = num_points_in_voxel<max_num_points_per_voxel?num_points_in_voxel:max_num_points_per_voxel;
  num_points_in_voxel = min(num_points_in_voxel, max_num_points_per_voxel);

  const auto voxel_idx = atomicAdd(num_filled_voxels, 1);
  num_points_per_voxel[voxel_idx] = num_points_in_voxel;

  const auto voxel_x = dense_voxel_idx % grid_x_size;
  const auto voxel_y = (dense_voxel_idx / grid_x_size) % grid_y_size;
  const auto voxel_z = dense_voxel_idx / (grid_y_size * grid_x_size);
  voxel_indices[voxel_idx] = make_int4(voxel_x, voxel_y, voxel_z, 0);

  for (int point_idx_in_voxel =0; point_idx_in_voxel<max_num_points_per_voxel; ++point_idx_in_voxel)
  {
    const auto in_idx = dense_voxel_idx * max_num_points_per_voxel + point_idx_in_voxel;
    const auto out_idx = voxel_idx * max_num_points_per_voxel + point_idx_in_voxel;
    voxel_features[out_idx] = point_idx_in_voxel < num_points_in_voxel ? dense_voxels[in_idx] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
}

void generate_base_features(
    const int* num_points_per_dense_voxel,
    const float* dense_voxels,
    int grid_x_size,
    int grid_y_size,
    int grid_z_size,
    int max_num_voxels,
    int max_num_points_per_voxel,
    int* num_filled_voxels,
    float* voxel_features,
    int* voxel_indices,
    int* num_points_per_voxel)
{
  constexpr int block_dim = 256;
  const int grid_dim = (max_num_voxels+block_dim-1)/block_dim;
  generate_base_features_kernel<<<grid_dim, block_dim, 0, nullptr>>>
    (num_points_per_dense_voxel,
    reinterpret_cast<const float4*>(dense_voxels),
    grid_x_size,
    grid_y_size,
    grid_z_size,
    max_num_voxels,
    max_num_points_per_voxel,
    num_filled_voxels,
    reinterpret_cast<float4*>(voxel_features),
    reinterpret_cast<int4*>(voxel_indices),
    num_points_per_voxel);
}
