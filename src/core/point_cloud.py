import open3d as o3d

from nptyping import Float, NDArray, Shape
from typing import Callable


class PointCloud:
    def __init__(
        self, path_to_pcd: str, getter: Callable[[str], o3d.geometry.PointCloud]
    ):
        self._path_to_pcd = path_to_pcd
        self._getter = getter

    def read(self) -> o3d.geometry.PointCloud:
        return self._getter(self._path_to_pcd)

    @staticmethod
    def voxel_down_sample(
        point_cloud: o3d.geometry.PointCloud,
        bounds: tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]],
        voxel_size: float,
    ):
        """
        Voxel down sampling with bounds given
        :param point_cloud: Point cloud for down sampling
        :param bounds: Bounds of the voxel grid
        :param voxel_size: Voxel size
        :return: Down sampled point cloud
        """
        min_bounds, max_bounds = bounds
        voxel_down_result, _, _ = point_cloud.voxel_down_sample_and_trace(
            voxel_size, min_bounds, max_bounds
        )
        return voxel_down_result
