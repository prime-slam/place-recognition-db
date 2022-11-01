import open3d as o3d

from typing import Callable


class PointCloudProvider:
    def __init__(
        self, path_to_pcd: str, getter: Callable[[str], o3d.t.geometry.PointCloud]
    ):
        self._path_to_pcd = path_to_pcd
        self._getter = getter

    @property
    def point_cloud(self) -> o3d.t.geometry.PointCloud:
        return self._getter(self._path_to_pcd)
