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
import random

from nptyping import Float, NDArray, Shape

from vprdb.core import Database


def generate_trajectory_from_positions(
    xyz_positions: list,
) -> list[NDArray[Shape["4, 4"], Float]]:
    poses = []
    for xyz in xyz_positions:
        pose = np.empty((4, 4))
        pose[:3, 3] = xyz
        poses.append(np.asarray(pose))
    return poses


def get_db_subset(database: Database, indices: list[int]) -> Database:
    new_rgb = [database.color_images[i] for i in indices]
    new_point_clouds = [database.point_clouds[i] for i in indices]
    new_traj = [database.trajectory[i] for i in indices]
    return Database(new_rgb, new_point_clouds, new_traj)


def generate_random_samples_of_test_db(number_of_samples: int):
    """Generates random-sized samples from database"""
    return [
        sorted(random.sample(range(5), k=random.randint(1, 5)))
        for _ in range(number_of_samples)
    ]
