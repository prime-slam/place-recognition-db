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
from vprdb.core import Database, VoxelGrid


def spatial_coverage(
    original_db: Database,
    reduced_db: Database,
    voxel_size: float = 0.3,
    down_sample_step: int = 100,
) -> float:
    """
    The metric determines how well the map from the original database is covered
    by the map from the reduced database

    :param original_db: Original database
    :param reduced_db: Reduced database
    :param voxel_size: Voxel size for down sampling
    :param down_sample_step: Voxel down sampling step for reducing RAM usage

    :return: the ratio of the number of points in the map from the reduced database
    after down sampling to the number of points in the map from the original database
    after down sampling
    """
    min_bounds, max_bounds = original_db.bounds
    voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)
    original_map = original_db.build_sparse_map(voxel_grid, down_sample_step)
    filtered_map = reduced_db.build_sparse_map(voxel_grid, down_sample_step)
    return len(filtered_map.points) / len(original_map.points)
