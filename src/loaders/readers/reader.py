import open3d as o3d

from abc import ABC, abstractmethod
from nptyping import NDArray, Shape, Float, UInt8
from src.core import Database
from src.loaders.providers import ImageProvider, PointCloudProvider


class Reader(ABC):
    def read_dataset(self) -> Database:
        rgb_images, pcds, trajectory = self._get_images_pcds_traj()
        return Database(trajectory, rgb_images, pcds)

    @abstractmethod
    def _get_images_pcds_traj(
        self,
    ) -> tuple[
        list[ImageProvider],
        list[PointCloudProvider],
        list[NDArray[Shape["4, 4"], Float]],
    ]:
        pass

    @abstractmethod
    def _get_rgb(self, path_to_image: str) -> NDArray[Shape["*, *, 3"], UInt8]:
        pass

    @abstractmethod
    def _get_pcd(self, path_to_pcd: str) -> o3d.t.geometry.PointCloud:
        pass
