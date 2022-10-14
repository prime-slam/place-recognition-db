import numpy as np

from src.core import Database
from src.reduction_methods.reduction_method import ReductionMethod


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
        x_min, y_min, z_min = min_over_axes
        length, width, height = (
            np.ceil((max_over_axes - min_over_axes) / self.cube_size) * self.cube_size
        )

        # Generating cubes
        def generate_segment(start, number_of_points):
            res_points = []
            for j in range(number_of_points):
                res_points.append(start + j * self.cube_size)
            return np.asarray(res_points)

        cubes_x = generate_segment(x_min, int(length / self.cube_size))
        cubes_y = generate_segment(y_min, int(width / self.cube_size))
        cubes_z = generate_segment(z_min, int(height / self.cube_size))

        # Association of points with cubes
        cubes = dict()
        for i, trans_matrix in enumerate(traj):
            point = trans_matrix[:3, 3]

            x, y, z = (
                cubes_x[cubes_x.searchsorted(point[0], side="right") - 1],
                cubes_y[cubes_y.searchsorted(point[1], side="right") - 1],
                cubes_z[cubes_z.searchsorted(point[2], side="right") - 1],
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

        new_traj = [db.trajectory[i] for i in res_indices]
        new_rgb = [db.images[i] for i in res_indices]
        new_pcds = [db.pcds[i] for i in res_indices]
        return Database(new_traj, new_rgb, new_pcds)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
