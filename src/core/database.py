import numpy as np
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

    def build_sparse_map(
        self,
        bounds: tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]],
        voxel_size: float,
        voxel_step: int = 100,
    ) -> o3d.geometry.PointCloud:
        """
        Builds sparse map of the whole DB scene
        :param bounds: Bounds of the scene
        :param voxel_size: Voxel size for down sampling
        :param voxel_step: Voxel down sampling step for reducing RAM usage
        :return: Resulting point cloud of the scene
        """
        global_pcd = o3d.geometry.PointCloud()
        for pose, pcd, i in zip(self.trajectory, self.pcds, range(len(self))):
            pcd = pcd.read().transform(pose)
            global_pcd += pcd
            if i % voxel_step == 0:
                global_pcd = PointCloud.voxel_down_sample(
                    global_pcd, bounds, voxel_size
                )
        return PointCloud.voxel_down_sample(global_pcd, bounds, voxel_size)

    def get_bounds(
        self,
    ) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
        """
        Gets bounds of the DB scene
        :return: Min and max bounds of the scene
        """
        min_bounds = np.empty((0, 3))
        max_bounds = np.empty((0, 3))

        for pose, pcd in zip(self.trajectory, self.pcds):
            pcd = pcd.read().transform(pose)
            min_bounds = np.append(min_bounds, [pcd.get_min_bound()], axis=0)
            max_bounds = np.append(max_bounds, [pcd.get_max_bound()], axis=0)

        min_bound = np.amin(min_bounds, axis=0)
        max_bound = np.amax(max_bounds, axis=0)
        return min_bound, max_bound
