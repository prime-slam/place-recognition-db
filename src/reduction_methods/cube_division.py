from math import floor, ceil, gcd
from src.reduction_methods.reduction_method import ReductionMethod
from src.database import Database

import numpy as np


class CubeDivision(ReductionMethod):
    """
    The method divides the cuboid in which the trajectory is located
    into many small cubes, in each of which only one pose is chosen
    """

    def __init__(self, cube_divider: int):
        """
        Constructs CubeDivision reduction method
        :param cube_divider: The side of the cube will be reduced by a factor of cube_divider
        """
        self.cube_divider = cube_divider

    def reduce(self, db: Database) -> Database:
        traj = np.asarray(db.trajectory)
        x_min, y_min, z_min = (
            floor(min(traj[:, 0, 3])),
            floor(min(traj[:, 1, 3])),
            floor(min(traj[:, 2, 3])),
        )
        x_max, y_max, z_max = (
            ceil(max(traj[:, 0, 3])),
            ceil(max(traj[:, 1, 3])),
            ceil(max(traj[:, 2, 3])),
        )

        length = x_max - x_min
        width = y_max - y_min
        height = z_max - z_min

        side = gcd(length, gcd(width, height)) / self.cube_divider

        # Generating cubes
        def generate_segment(start, number_of_points):
            res_points = []
            for j in range(number_of_points):
                res_points.append(start + j * side)
            return res_points

        cubes_x = generate_segment(x_min, length // side)
        cubes_y = generate_segment(y_min, width // side)
        cubes_z = generate_segment(z_min, height // side)

        # Association of points with cubes
        cubes = dict()
        for i, trans_matrix in enumerate(traj):
            point = trans_matrix[:3, 3]

            def find_cube_coordinate(axis, coordinate):
                res = None
                for j in range(len(axis)):
                    if axis[j] <= coordinate:
                        res = axis[j]
                    else:
                        break
                return res

            x, y, z = (
                find_cube_coordinate(cubes_x, point[0]),
                find_cube_coordinate(cubes_y, point[1]),
                find_cube_coordinate(cubes_z, point[2]),
            )
            point = np.append(point, i)
            if (x, y, z) in cubes:
                cubes[(x, y, z)] = np.append(cubes[(x, y, z)], [point], axis=0)
            else:
                cubes[(x, y, z)] = np.asarray([point])

        # Finding the closest point to the center in each cube
        res_indices = []
        for (x, y, z), enum_points in cubes.items():
            indices = enum_points[:, 3]
            points = enum_points[:, :3]
            cube_center = (x + side / 2, y + side / 2, z + side / 2)
            distances = np.apply_along_axis(
                lambda p: np.linalg.norm(p - cube_center), axis=1, arr=points
            )
            res_indices.append(int(indices[np.argmin(distances)]))
        res_indices.sort()

        new_traj = [db.trajectory[i] for i in res_indices]
        new_rgb = [db.rgb_images_paths[i] for i in res_indices]
        new_pcds = [db.pcds_paths[i] for i in res_indices]
        return Database(new_traj, new_rgb, new_pcds, db.rgb_getter, db.pcd_getter)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
