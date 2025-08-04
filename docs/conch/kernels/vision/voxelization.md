# Voxelization

Given a set of points in 3D space, assign each point to a voxel.

Args:
- points: Tensor[N, 3 + C], 
- voxel

Returns:
1. The corresponding voxel coordinates for each point (easy, dynamic)
2. Grouped points by voxel


dense_voxelized_points = Tensor[num_voxels, max_num_points_per_voxel, 3 + c]

What are the coordinates of each voxel?

How many points are in each voxel?
