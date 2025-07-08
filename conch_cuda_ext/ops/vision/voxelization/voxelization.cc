// Copyright (c) OpenMMLab. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

int HardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

int NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3);

void DynamicVoxelizeForwardCUDAKernelLauncher(
    const at::Tensor &points, at::Tensor &coors,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const int NDim = 3);

int hard_voxelize_forward_impl(const at::Tensor &points, at::Tensor &voxels,
                               at::Tensor &coors,
                               at::Tensor &num_points_per_voxel,
                               const std::vector<float> voxel_size,
                               const std::vector<float> coors_range,
                               const int max_points, const int max_voxels,
                               const int NDim = 3) {
    return HardVoxelizeForwardCUDAKernelLauncher(points, voxels, coors,
                              num_points_per_voxel, voxel_size, coors_range,
                              max_points, max_voxels, NDim);
}

int nondeterministic_hard_voxelize_forward_impl(
    const at::Tensor &points, at::Tensor &voxels, at::Tensor &coors,
    at::Tensor &num_points_per_voxel, const std::vector<float> voxel_size,
    const std::vector<float> coors_range, const int max_points,
    const int max_voxels, const int NDim = 3) {
    return NondeterministicHardVoxelizeForwardCUDAKernelLauncher(
                              points, voxels, coors, num_points_per_voxel,
                              voxel_size, coors_range, max_points, max_voxels,
                              NDim);
}

void dynamic_voxelize_forward_impl(const at::Tensor &points, at::Tensor &coors,
                                   const std::vector<float> voxel_size,
                                   const std::vector<float> coors_range,
                                   const int NDim = 3) {
    DynamicVoxelizeForwardCUDAKernelLauncher(points, coors, voxel_size,
                       coors_range, NDim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hard_voxelize_forward", &hard_voxelize_forward_impl,
        "hard_voxelize_forward");
  m.def("nondeterministic_hard_voxelize_forward", &nondeterministic_hard_voxelize_forward_impl,
        "nondeterministic_hard_voxelize_forward");
  m.def("dynamic_voxelize_forward", &dynamic_voxelize_forward_impl,
        "dynamic_voxelize_forward");
}
