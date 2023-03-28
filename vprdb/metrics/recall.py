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

from vprdb.core import calculate_point_cloud_coverage, Database, VoxelGrid


def recall(
    source_db: Database,
    test_db: Database,
    matches: list[int],
    voxel_size: float = 0.3,
    threshold: float = 0.3,
) -> float:
    """
    The metric finds the number of correctly matched frames

    :param source_db: Database for the VPR task
    :param test_db: Database used as queries to the VPR system
    :param matches: VPR system matches.
    The i-th value from the list is the index of the frame from the database,
    which will be matched with the i-th test frame
    :param voxel_size: Voxel size for down sampling
    :param threshold: The value of frame coverage,
    below which the frame will be considered uncovered

    :return: Recall value
    """
    if len(test_db) != len(matches):
        raise ValueError(
            "The length of the matches and the test database must be the same"
        )

    min_bounds_test, max_bounds_test = test_db.bounds
    min_bounds_source, max_bounds_source = source_db.bounds
    min_bounds = np.amin(np.row_stack((min_bounds_test, min_bounds_source)), axis=0)
    max_bounds = np.amax(np.row_stack((max_bounds_test, max_bounds_source)), axis=0)

    voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)
    results = []
    for i, match in enumerate(matches):
        pose_query = test_db.trajectory[i]
        pcd_query = test_db.point_clouds[i].point_cloud.transform(pose_query)

        pose_source = source_db.trajectory[match]
        pcd_source = source_db.point_clouds[match].point_cloud.transform(pose_source)

        coverage = calculate_point_cloud_coverage(pcd_query, pcd_source, voxel_grid)
        results.append(coverage > threshold)

    return sum(results) / len(results)
