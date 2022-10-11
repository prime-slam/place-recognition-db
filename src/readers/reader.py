import open3d as o3d
from abc import ABC, abstractmethod
from nptyping import NDArray, Shape, Float, UInt8
from src.database import Database


class Reader(ABC):
    @abstractmethod
    def read_dataset(self) -> Database:
        rgb_images, pcds, trajectory = self._get_images_pcds_traj()
        get_rgb_image, get_pcd = self._rgb_getter, self._pcd_getter
        return Database(trajectory, rgb_images, pcds, get_rgb_image, get_pcd)

    @abstractmethod
    def _get_images_pcds_traj(
        self,
    ) -> tuple[list[str], list[str], list[NDArray[Shape["4, 4"], Float]]]:
        pass

    @staticmethod
    @abstractmethod
    def _rgb_getter(path_to_image: str) -> NDArray[Shape["*, *, 3"], UInt8]:
        pass

    @staticmethod
    @abstractmethod
    def _pcd_getter(path_to_pcd: str) -> o3d.geometry.PointCloud:
        pass
