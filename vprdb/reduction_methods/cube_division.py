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

from vprdb.core import Database, VoxelGrid
from vprdb.reduction_methods.reduction_method import ReductionMethod


class CubeDivision(ReductionMethod):
    """
    The method divides the cuboid in which the trajectory is located
    into many small cubes, in each of which only one pose is chosen
    """

    def __init__(self, cube_size: float):
        """
        Constructs CubeDivision reduction method
        :param cube_size: The size of the cubes into which the cuboid will be divided
        """
        self.cube_size = cube_size

    def reduce(self, db: Database) -> Database:
        traj = np.asarray(db.trajectory)
        xyz_traj = traj[:, :3, 3]
        min_over_axes = np.amin(xyz_traj, axis=0)
        max_over_axes = np.amax(xyz_traj, axis=0)
        voxel_grid = VoxelGrid(min_over_axes, max_over_axes, self.cube_size)

        # Association of points with cubes
        cubes = dict()
        for i, trans_matrix in enumerate(traj):
            point = trans_matrix[:3, 3]
            cube_coordinates = voxel_grid.get_voxel_coordinates(point)

            point = np.append(point, i)
            if cube_coordinates in cubes:
                cubes[cube_coordinates] = np.append(
                    cubes[cube_coordinates], [point], axis=0
                )
            else:
                cubes[cube_coordinates] = np.asarray([point])

        # Finding the closest point to the center in each cube
        res_indices = []
        for (x, y, z), enum_points in cubes.items():
            indices = enum_points[:, 3]
            points = enum_points[:, :3]
            cube_center = (
                x + self.cube_size / 2,
                y + self.cube_size / 2,
                z + self.cube_size / 2,
            )
            distances = np.apply_along_axis(
                lambda p: np.linalg.norm(p - cube_center), axis=1, arr=points
            )
            res_indices.append(int(indices[np.argmin(distances)]))
        res_indices.sort()

        new_rgb = [db.color_images[i] for i in res_indices]
        new_point_clouds = [db.point_clouds[i] for i in res_indices]
        new_traj = [db.trajectory[i] for i in res_indices]
        return Database(new_rgb, new_point_clouds, new_traj)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
