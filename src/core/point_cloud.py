import open3d as o3d

from typing import Callable


class PointCloud:
    def __init__(
        self, path_to_pcd: str, getter: Callable[[str], o3d.geometry.PointCloud]
    ):
        self._path_to_pcd = path_to_pcd
        self._getter = getter

    def read(self) -> o3d.geometry.PointCloud:
        return self._getter(self._path_to_pcd)
