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

from vprdb.core import Database
from vprdb.reduction_methods.reduction_method import ReductionMethod


class DistanceVector(ReductionMethod):
    """
    The method calculates the distance between neighboring poses.
    Then using the calculated distances, it calculates the distance from the current pose to the last pose taken.
    If the distance is greater than the threshold value, the current pose is added to the final database.
    """

    def __init__(self, distance_threshold: float):
        """
        Constructs DistanceVector reduction method
        :param distance_threshold: Threshold value for distance
        """
        self.distance_threshold = distance_threshold

    def reduce(self, db: Database) -> Database:
        traj = np.asarray(db.trajectory)
        new_traj = [traj[0]]
        new_color = [db.color_images[0]]
        new_point_clouds = [db.point_clouds[0]]
        first_points = traj[:-1, :3, 3]
        last_points = traj[1:, :3, 3]
        distances = np.linalg.norm(last_points - first_points, axis=1)
        partial_distance = 0
        for i, cur_distance in enumerate(distances):
            partial_distance += cur_distance
            if partial_distance > self.distance_threshold:
                new_traj.append(traj[i + 1])
                new_color.append(db.color_images[i + 1])
                new_point_clouds.append(db.point_clouds[i + 1])
                partial_distance = 0

        return Database(new_color, new_point_clouds, new_traj)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
