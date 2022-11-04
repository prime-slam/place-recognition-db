import open3d as o3d

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PointCloudProvider:
    path_to_pcd: str
    getter: Callable[[str], o3d.t.geometry.PointCloud]

    @property
    def point_cloud(self) -> o3d.t.geometry.PointCloud:
        return self.getter(self.path_to_pcd)

    def __hash__(self):
        return hash(self.path_to_pcd)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.path_to_pcd == other.path_to_pcd

    # Method for proper joblib hashing based on pickle
    def __getstate__(self):
        return self.path_to_pcd
