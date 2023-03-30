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
import numpy as np
import open3d as o3d

from nptyping import Float, NDArray, Shape

from vprdb.core.database import Database
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


def calculate_iou(
    pcd_1: o3d.geometry.PointCloud,
    pcd_2: o3d.geometry.PointCloud,
    voxel_grid: VoxelGrid,
) -> float:
    """
    Calculates IoU for two point clouds
    :param pcd_1: First point cloud
    :param pcd_2: Second point cloud
    :param voxel_grid: Voxel grid for down sampling
    :return: IoU ∈ [0; 1]
    """
    pcd_1 = voxel_grid.voxel_down_sample(pcd_1)
    pcd_2 = voxel_grid.voxel_down_sample(pcd_2)
    united_map = voxel_grid.voxel_down_sample(pcd_1 + pcd_2)

    united_map_size = len(united_map.points)
    intersection_size = len(pcd_1.points) + len(pcd_2.points) - united_map_size
    iou = intersection_size / united_map_size
    return iou


def match_two_databases(
    source_db: Database, target_db: Database, voxel_grid: VoxelGrid
) -> list[int]:
    """
    Builds matches between two databases. Databases should have the same coordinate system
    :param source_db: The database for which matches are being built
    :param target_db: The database on the base of which the matches will be built
    :param voxel_grid: VoxelGrid for down sampling
    :return: Output matches.
    The i-th value from the list is the index of the frame from the target database,
    which is matched with the i-th source frame
    """
    matches = []
    for pose, pcd_raw in zip(source_db.trajectory, source_db.point_clouds):
        cur_coverages = []
        for target_pose, target_pcd in zip(
            target_db.trajectory, target_db.point_clouds
        ):
            pcd_query = pcd_raw.point_cloud.transform(pose)
            pcd_db = target_pcd.point_cloud.transform(target_pose)
            coverage = calculate_point_cloud_coverage(pcd_query, pcd_db, voxel_grid)
            cur_coverages.append(coverage)
        best_match = np.argmax(cur_coverages)
        matches.append(int(best_match))
    return matches


def find_bounds_for_multiple_databases(
    databases: list[Database],
) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
    """
    Finds bounds for multiple databases
    :param databases: List of databases for finding bounds
    :return: Min and max bounds
    """
    min_bounds_global, max_bounds_global = [], []
    for db in databases:
        min_bounds, max_bounds = db.bounds
        min_bounds_global.append(min_bounds)
        max_bounds_global.append(max_bounds)
    min_bounds_global, max_bounds_global = np.asarray(min_bounds_global), np.asarray(
        max_bounds_global
    )
    min_bound = np.amin(min_bounds_global, axis=0)
    max_bound = np.amax(max_bounds_global, axis=0)
    return min_bound, max_bound
