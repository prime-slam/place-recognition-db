import mrob
import numpy as np
import os

from nptyping import Float, NDArray, Shape

from src.core import Database
from src.providers import ColorImageProvider, PointCloudProvider


def read_trajectory(path_to_traj: str) -> list[NDArray[Shape["4, 4"], Float]]:
    poses_quat = []
    with open(path_to_traj, "r") as file:
        for line in file:
            poses_quat.append([float(i) for i in line.split(" ")])

    Ts = []
    for i, pose in enumerate(poses_quat):
        t = pose[1:4]
        R = mrob.geometry.quat_to_so3(pose[4:8])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        Ts.append(T)
    return Ts


def read_folder(folder_path: str) -> list[str]:
    return sorted(os.listdir(folder_path))


def read_dataset(
    path_to_dataset: str,
    pcd_folder_name: str = "pcd",
    rgb_folder_name: str = "rgb",
    traj_file_name: str = "groundtruth.txt",
) -> Database:
    path_to_pcds = os.path.join(path_to_dataset, pcd_folder_name)
    path_to_rgb_images = os.path.join(path_to_dataset, rgb_folder_name)
    point_clouds = [
        PointCloudProvider(os.path.join(path_to_pcds, pcd_file_name))
        for pcd_file_name in read_folder(path_to_pcds)
    ]
    rgb_images = [
        ColorImageProvider(os.path.join(path_to_rgb_images, rgb_file_name))
        for rgb_file_name in read_folder(path_to_rgb_images)
    ]
    trajectory = read_trajectory(os.path.join(path_to_dataset, traj_file_name))
    return Database(trajectory, rgb_images, point_clouds)
