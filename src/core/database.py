import open3d as o3d

from dataclasses import dataclass
from nptyping import Float, NDArray, Shape, UInt8
from src.core.image import Image
from src.core.point_cloud import PointCloud


@dataclass(frozen=True)
class Database:
    trajectory: list[NDArray[Shape["4, 4"], Float]]
    images: list[Image]
    pcds: list[PointCloud]

    def __post_init__(self):
        if not (len(self.trajectory) == len(self.images) == len(self.pcds)):
            raise ValueError("Trajectory, RGB images and PCDs should have equal length")

    def __len__(self):
        return len(self.trajectory)

    def get_rgb_image_by_index(self, n: int) -> NDArray[Shape["*, *, 3"], UInt8]:
        return self.images[n].read()

    def get_pcd_by_index(self, n: int) -> o3d.geometry.PointCloud:
        return self.pcds[n].read()
