#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import open3d as o3d

from vprdb.core.voxel_grid import VoxelGrid


def calculate_point_cloud_coverage(
    query_pcd: o3d.geometry.PointCloud,
    db_pcd: o3d.geometry.PointCloud,
    voxel_grid: VoxelGrid,
) -> float:
    """
    Calculates coverage for query point cloud by database point cloud
    :param query_pcd: query point cloud
    :param db_pcd: database point cloud
    :param voxel_grid: voxel grid for down sampling
    :return: coverage ∈ [0; 1]
    """
    query_pcd = voxel_grid.voxel_down_sample(query_pcd)
    db_pcd = voxel_grid.voxel_down_sample(db_pcd)
    united_map = voxel_grid.voxel_down_sample(query_pcd + db_pcd)

    query_size = len(query_pcd.points)
    intersection_size = query_size + len(db_pcd.points) - len(united_map.points)
    return intersection_size / query_size
