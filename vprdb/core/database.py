import numpy as np

from dataclasses import dataclass
from functools import cached_property
from nptyping import Float, NDArray, Shape
from pathlib import Path

from vprdb.providers import ColorImageProvider, DepthImageProvider, PointCloudProvider


@dataclass(frozen=True)
class Database:
    color_images: list[ColorImageProvider]
    spatial_items: list[DepthImageProvider | PointCloudProvider]
    trajectory: list[NDArray[Shape["4, 4"], Float]]

    def __post_init__(self):
        if not (
            len(self.trajectory) == len(self.spatial_items) == len(self.trajectory)
        ):
            raise ValueError(
                "Trajectory, RGB images and spatial items should have equal length"
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
            pcd = self.spatial_items[i].point_cloud.transform(self.trajectory[i])
            min_bounds.append(pcd.get_min_bound())
            max_bounds.append(pcd.get_max_bound())

        min_bound = np.amin(np.asarray(min_bounds), axis=0)
        max_bound = np.amax(np.asarray(max_bounds), axis=0)
        return min_bound, max_bound
