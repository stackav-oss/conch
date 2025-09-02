// Copyright 2025 Stack AV Co.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void generate_dense_voxels(
    const float* points,
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
    int* num_points_per_dense_voxel,
    float* dense_voxels);

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
    int* num_points_per_voxel);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> generate_voxels(
  const at::Tensor points,
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
  int max_num_voxels) {
  const auto num_points = static_cast<int>(points.size(0));
  const auto num_features_per_point = points.size(1);
  assert(num_features_per_point == 4);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(points));

  const auto float_tensor_options = torch::TensorOptions().dtype(points.dtype()).device(points.device());
  const auto int_tensor_options = torch::TensorOptions().dtype(torch::kInt32).device(points.device());

  // atomic counters init to 0
  at::Tensor num_filled_voxels = torch::zeros({1}, int_tensor_options);
  at::Tensor num_points_per_dense_voxel = torch::zeros({max_num_voxels}, int_tensor_options);

  at::Tensor dense_voxels = torch::empty({max_num_voxels, max_num_points_per_voxel, num_features_per_point}, float_tensor_options);
  at::Tensor voxel_features = torch::empty({max_num_voxels, max_num_points_per_voxel, num_features_per_point}, float_tensor_options);
  at::Tensor voxel_indices = torch::empty({max_num_voxels, 4}, int_tensor_options);
  at::Tensor num_points_per_voxel = torch::empty({max_num_voxels}, int_tensor_options);

  generate_dense_voxels(
    points.data_ptr<float>(),
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
    num_points_per_dense_voxel.data_ptr<int>(),
    dense_voxels.data_ptr<float>());

  generate_base_features(
        num_points_per_dense_voxel.data_ptr<int>(),
        dense_voxels.data_ptr<float>(),
        grid_x_size,
        grid_y_size,
        grid_z_size,
        max_num_voxels,
        max_num_points_per_voxel,
        num_filled_voxels.data_ptr<int>(),
        voxel_features.data_ptr<float>(),
        voxel_indices.data_ptr<int>(),
        num_points_per_voxel.data_ptr<int>());

  // synchronize
  const auto num_filled_voxels_cpu = num_filled_voxels.cpu();
  return {num_filled_voxels_cpu, voxel_features, voxel_indices, num_points_per_voxel};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_voxels", &generate_voxels,
        "generate_voxels");
}
