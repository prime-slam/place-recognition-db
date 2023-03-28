import numpy as np
import open3d as o3d

from dataclasses import dataclass
from functools import cached_property
from nptyping import Float, NDArray, Shape
from pathlib import Path

from vprdb.core.voxel_grid import VoxelGrid
from vprdb.providers import ColorImageProvider, DepthImageProvider, PointCloudProvider


@dataclass(frozen=True)
class Database:
    color_images: list[ColorImageProvider]
    point_clouds: list[DepthImageProvider | PointCloudProvider]
    trajectory: list[NDArray[Shape["4, 4"], Float]]

    def __post_init__(self):
        if not (len(self.trajectory) == len(self.point_clouds) == len(self.trajectory)):
            raise ValueError(
                "Trajectory, RGB images and point clouds should have equal length"
            )

    @classmethod
    def from_depth_images(
        cls,
        color_images_paths: list[Path],
        depth_images_paths: list[Path],
        depth_scale: int,
        intrinsics: NDArray[Shape["3, 3"], Float],
        trajectory: list[NDArray[Shape["4, 4"], Float]],
    ):
        color_images_providers = [
            ColorImageProvider(path_to_image) for path_to_image in color_images_paths
        ]
        depth_images_providers = [
            DepthImageProvider(path_to_image, intrinsics, depth_scale)
            for path_to_image in depth_images_paths
        ]
        return cls(color_images_providers, depth_images_providers, trajectory)

    @classmethod
    def from_point_clouds(
        cls,
        color_images_paths: list[Path],
        point_clouds_paths: list[Path],
        trajectory: list[NDArray[Shape["4, 4"], Float]],
    ):
        color_images_providers = [
            ColorImageProvider(path_to_image) for path_to_image in color_images_paths
        ]
        point_clouds_providers = [
            PointCloudProvider(path_to_pcd) for path_to_pcd in point_clouds_paths
        ]
        return cls(color_images_providers, point_clouds_providers, trajectory)

    def __len__(self):
        return len(self.trajectory)

    @cached_property
    def bounds(
        self,
    ) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
        """
        Gets bounds of the DB scene
        :return: Min and max bounds of the scene
        """
        min_bounds = []
        max_bounds = []

        for i in range(len(self)):
            pcd = self.point_clouds[i].point_cloud.transform(self.trajectory[i])
            min_bounds.append(pcd.get_min_bound())
            max_bounds.append(pcd.get_max_bound())

        min_bound = np.amin(np.asarray(min_bounds), axis=0)
        max_bound = np.amax(np.asarray(max_bounds), axis=0)
        return min_bound, max_bound

    def build_sparse_map(
        self,
        voxel_grid: VoxelGrid,
        down_sample_step: int,
    ) -> o3d.geometry.PointCloud:
        """
        Builds sparse map of the whole DB scene
        :param voxel_grid: Voxel grid for down sampling
        :param down_sample_step: Voxel down sampling step for reducing RAM usage
        :return: Resulting point cloud of the scene
        """
        map_pcd = o3d.geometry.PointCloud()
        for i, (pose, pcd_raw) in enumerate(zip(self.trajectory, self.point_clouds)):
            pcd = pcd_raw.point_cloud.transform(pose)
            map_pcd += pcd
            if i % down_sample_step == 0:
                map_pcd = voxel_grid.voxel_down_sample(map_pcd)
        return voxel_grid.voxel_down_sample(map_pcd)
