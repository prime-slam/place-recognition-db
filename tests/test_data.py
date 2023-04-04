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

from pathlib import Path

from tests.utils import generate_trajectory_from_positions
from vprdb.core import Database
from vprdb.io import read_dataset_from_depth

artificial_traj_1 = generate_trajectory_from_positions(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 1, 1], [4, 1, 1]]
)
artificial_traj_2 = generate_trajectory_from_positions(
    [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
)

path_to_db = Path("tests/test_db")
intrinsics = np.loadtxt(path_to_db / "intrinsics.txt")
real_db = read_dataset_from_depth(
    path_to_db,
    color_dir="color",
    depth_dir="depth",
    trajectory_file_name="poses.txt",
    intrinsics=intrinsics,
    depth_scale=1000,
    with_timestamps=False,
)
artificial_db_1 = Database(
    real_db.color_images, real_db.point_clouds, artificial_traj_1
)
artificial_db_2 = Database(
    real_db.color_images, real_db.point_clouds, artificial_traj_2
)
