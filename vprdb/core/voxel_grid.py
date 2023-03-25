import numpy as np
import open3d as o3d

from dataclasses import dataclass
from nptyping import Float, NDArray, Shape


@dataclass
class VoxelGrid:
    """Voxel grid with given boundaries and voxel size"""

    min_bounds: NDArray[Shape["3"], Float]
    max_bounds: NDArray[Shape["3"], Float]
    voxel_size: float

    def get_voxel_index(
        self, point: NDArray[Shape["3"], Float]
    ) -> tuple[int, int, int]:
        """
        The method gets the voxel index for a given point
        Implemented according to the corresponding Open3D method
        :param point: Point to get its corresponding voxel index
        :return: Voxel index
        """
        ref_coord = (point - self.min_bounds) / self.voxel_size
        x_index, y_index, z_index = np.floor(ref_coord).astype(int)
        return x_index, y_index, z_index

    def get_voxel_coordinates(
        self, point: NDArray[Shape["3"], Float]
    ) -> tuple[float, float, float]:
        """
        The method gets the voxel coordinates for a given point
        Implemented according to the corresponding Open3D method
        :param point: Point to get its corresponding voxel coordinates
        :return: Voxel coordinates
        """
        return tuple(
            self.min_bounds
            + (np.asarray(self.get_voxel_index(point)) * self.voxel_size)
        )

    def voxel_down_sample(
        self, point_cloud: o3d.geometry.PointCloud
    ) -> o3d.geometry.PointCloud:
        """
        Voxel down sampling with bounds given
        :param point_cloud: Point cloud for down sampling
        :return: Down sampled point cloud
        """
        voxel_down_result, _, _ = point_cloud.voxel_down_sample_and_trace(
            self.voxel_size, self.min_bounds, self.max_bounds
        )
        return voxel_down_result
