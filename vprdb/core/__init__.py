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
"""
`core` submodule contains a Database class for easy storage and processing of
color images, depth images and trajectory and their further use for the VPR task.

`VoxelGrid` and `utils` provide various operations on point clouds.
"""
from vprdb.core.database import Database
from vprdb.core.utils import (
    calculate_iou,
    calculate_point_cloud_coverage,
    find_bounds_for_multiple_databases,
    match_two_databases,
)
from vprdb.core.voxel_grid import VoxelGrid
