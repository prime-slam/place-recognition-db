import numpy as np
import os
from typing import Callable

from nptyping import NDArray, Shape, Float, UInt8
import open3d as o3d
import cv2


class Database:
    def __init__(
        self,
        trajectory: list[NDArray[Shape["4, 4"], Float]],
        rgb_images_paths: list[str],
        pcds_paths: list[str],
        rgb_getter: Callable[[str], NDArray[Shape["*, *, 3"], UInt8]],
        pcd_getter: Callable[[str], o3d.geometry.PointCloud],
    ):
        if not (len(trajectory) == len(rgb_images_paths) == len(pcds_paths)):
            raise ValueError("Trajectory, RGB images and PCDs should have equal length")
        self._trajectory = trajectory
        self._rgb_images_paths = rgb_images_paths
        self._pcds_paths = pcds_paths
        self.rgb_getter = rgb_getter
        self.pcd_getter = pcd_getter

    def __len__(self):
        return len(self.trajectory)

    @property
    def trajectory(self) -> list[NDArray[Shape["4, 4"], Float]]:
        return self._trajectory

    @property
    def rgb_images_paths(self) -> list[str]:
        return self._rgb_images_paths

    @property
    def pcds_paths(self) -> list[str]:
        return self._pcds_paths

    def get_rgb_image_by_index(self, n: int) -> NDArray[Shape["*, *, 3"], UInt8]:
        return self.rgb_getter(self._rgb_images_paths[n])

    def get_pcd_by_index(self, n: int) -> o3d.geometry.PointCloud:
        return self.pcd_getter(self._pcds_paths[n])

    def export(self, path_to_save: str):
        path_to_save = os.path.join(path_to_save, "exported_data")
        if os.path.isdir(path_to_save):
            # TODO: change exception type
            raise Exception("exported_data directory exists")
        os.mkdir(path_to_save)
        path_to_traj = os.path.join(path_to_save, "trajectory.txt")
        with open(path_to_traj, "a") as traj:
            for pose in self.trajectory:
                R = pose[:3, :3].astype(np.str)
                t = pose[:3, 3].astype(np.str)
                flatten_pose = np.concatenate((R.flatten(), t))
                string_pose = " ".join(flatten_pose)
                traj.write(f"{string_pose}\n")
        path_to_rgb = os.path.join(path_to_save, "rgb")
        path_to_pcds = os.path.join(path_to_save, "pcds")
        os.mkdir(path_to_rgb)
        os.mkdir(path_to_pcds)
        for i in range(len(self)):
            image = self.get_rgb_image_by_index(i)
            cv2.imwrite(os.path.join(path_to_rgb, f"{i}.png"), image)
            pcd = self.get_pcd_by_index(i)
            o3d.io.write_point_cloud(os.path.join(path_to_pcds, f"{i}.pcd"), pcd)
