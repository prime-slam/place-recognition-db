import open3d as o3d

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PointCloudProvider:
    path: Path

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
        return o3d.io.read_point_cloud(str(self.path))
