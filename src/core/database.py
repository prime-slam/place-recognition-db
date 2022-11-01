import numpy as np
import open3d as o3d

from dataclasses import dataclass
from functools import cached_property
from nptyping import Float, NDArray, Shape, UInt8
from src.core.cache import memory
from src.core.image_provider import ImageProvider
from src.core.point_cloud_provider import PointCloudProvider
from src.core.utils import voxel_down_sample
from src.core.voxel_grid import VoxelGrid


@memory.cache
def _build_sparse_map(
    trajectory: list[NDArray[Shape["4, 4"], Float]],
    pcds: list[PointCloudProvider],
    voxel_grid: VoxelGrid,
    down_sample_step: int = 100,
) -> o3d.t.geometry.PointCloud:
    global_pcd = o3d.t.geometry.PointCloud()
    global_pcd.point.positions = o3d.core.Tensor.empty((0, 3))
    for i, (pose, pcd_raw) in enumerate(zip(trajectory, pcds)):
        pcd = pcd_raw.point_cloud.transform(pose)
        global_pcd += pcd
        if i % down_sample_step == 0:
            global_pcd = voxel_down_sample(global_pcd, voxel_grid)
    return voxel_down_sample(global_pcd, voxel_grid)


@dataclass(frozen=True)
class Database:
    trajectory: list[NDArray[Shape["4, 4"], Float]]
    images: list[ImageProvider]
    pcds: list[PointCloudProvider]

    def __post_init__(self):
        if not (len(self.trajectory) == len(self.images) == len(self.pcds)):
            raise ValueError("Trajectory, RGB images and PCDs should have equal length")

    def __len__(self):
        return len(self.trajectory)

    def get_rgb_image_by_index(self, n: int) -> NDArray[Shape["*, *, 3"], UInt8]:
        return self.images[n].image

    def get_pcd_by_index(self, n: int) -> o3d.geometry.PointCloud:
        return self.pcds[n].point_cloud

    def build_sparse_map(
        self,
        voxel_grid: VoxelGrid,
        down_sample_step: int = 100,
    ) -> o3d.t.geometry.PointCloud:
        """
        Builds sparse map of the whole DB scene
        :param voxel_grid: Voxel grid for down_sampling
        :param down_sample_step: Voxel down sampling step for reducing RAM usage
        :return: Resulting point cloud of the scene
        """
        return _build_sparse_map(
            self.trajectory, self.pcds, voxel_grid, down_sample_step
        )

    @cached_property
    def bounds(
        self,
    ) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
        """
        Gets bounds of the DB scene
        :return: Min and max bounds of the scene
        """
        min_bounds = np.empty((0, 3))
        max_bounds = np.empty((0, 3))

        for pose, pcd_raw in zip(self.trajectory, self.pcds):
            pcd = pcd_raw.point_cloud.transform(pose)
            min_bounds = np.append(min_bounds, [pcd.get_min_bound().numpy()], axis=0)
            max_bounds = np.append(max_bounds, [pcd.get_max_bound().numpy()], axis=0)

        min_bound = np.amin(min_bounds, axis=0)
        max_bound = np.amax(max_bounds, axis=0)
        return min_bound, max_bound
