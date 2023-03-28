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
import os

from joblib import Parallel, delayed

from vprdb.core import calculate_point_cloud_coverage, Database, VoxelGrid


def frames_coverage(
    original_db: Database,
    reduced_db: Database,
    voxel_size: float = 0.3,
    num_of_threads: int = os.cpu_count(),
) -> list[float]:
    """
    The metric determines how well the frames from the original database
    are covered by the frames from the reduced database

    :param original_db: Original database
    :param reduced_db: Reduced database
    :param voxel_size: Voxel size for down sampling
    :param num_of_threads: Number of threads to parallelize calculations

    :return: A list of values indicating the level of coverage of a particular frame
    """
    min_bounds, max_bounds = original_db.bounds
    voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)

    def find_best_coverage(pose, pcd_raw):
        coverages_for_query = []
        for reduced_db_pose, reduced_db_pcd in zip(
            reduced_db.trajectory, reduced_db.point_clouds
        ):
            pcd_query = pcd_raw.point_cloud.transform(pose)
            pcd_db = reduced_db_pcd.point_cloud.transform(reduced_db_pose)
            coverage = calculate_point_cloud_coverage(pcd_query, pcd_db, voxel_grid)
            coverages_for_query.append(coverage)
        return max(coverages_for_query)

    coverages = Parallel(n_jobs=num_of_threads)(
        delayed(find_best_coverage)(pose, pcd_raw)
        for pose, pcd_raw in zip(original_db.trajectory, original_db.point_clouds)
    )
    return coverages
