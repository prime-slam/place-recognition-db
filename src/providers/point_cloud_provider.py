import open3d as o3d

from dataclasses import dataclass


@dataclass(frozen=True)
class PointCloudProvider:
    path_to_pcd: str

    @property
    def point_cloud(self) -> o3d.t.geometry.PointCloud:
        return o3d.t.io.read_point_cloud(self.path_to_pcd)

    def __hash__(self):
        return hash(self.path_to_pcd)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.path_to_pcd == other.path_to_pcd
