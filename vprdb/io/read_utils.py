import mrob
import numpy as np

from nptyping import Float, NDArray, Shape
from pathlib import Path

from vprdb.core import Database


def __load_poses(poses_path: Path, with_timestamps: bool):
    poses_quat = []
    with open(poses_path, "r") as file:
        for line in file:
            poses_quat.append([float(i) for i in line.strip().split(" ")])

    poses = []
    for pose in poses_quat:
        t = pose[1:4] if with_timestamps else pose[:3]
        R = pose[4:8] if with_timestamps else pose[3:7]
        R = mrob.geometry.quat_to_so3(R)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses.append(T)

    return poses


def __read_dir(dir_path: Path) -> list[Path]:
    return sorted(list(dir_path.iterdir()))


def read_dataset_from_depth(
    path_to_dataset: Path,
    color_dir: str,
    depth_dir: str,
    trajectory_file_name: str,
    intrinsics: NDArray[Shape["3, 3"], Float],
    depth_scale: int,
    with_timestamps: bool = True,
) -> Database:
    """
    Reads dataset from given directory
    :param path_to_dataset: The name of the directory to read
    :param color_dir: The name of the directory with color images
    :param depth_dir: The name of the directory with depth images
    :param trajectory_file_name: The name of file with trajectory
    :param intrinsics: NumPy array with camera intrinsics
    :param depth_scale: Depth scale
    :param with_timestamps: Indicates that a trajectory with timestamps is given
    """
    rgb_images = __read_dir(path_to_dataset / color_dir)
    depth_images = __read_dir(path_to_dataset / depth_dir)

    traj = __load_poses(path_to_dataset / trajectory_file_name, with_timestamps)

    database = Database.from_depth_images(
        rgb_images, depth_images, depth_scale, intrinsics, traj
    )
    return database


def read_dataset_from_point_clouds(
    path_to_dataset: Path,
    color_dir: str,
    point_clouds_dir: str,
    trajectory_file_name: str,
    with_timestamps: bool = True,
) -> Database:
    """
    Reads dataset from given directory
    :param path_to_dataset: The name of the directory to read
    :param color_dir: The name of the directory with color images
    :param point_clouds_dir: The name of the directory with point clouds
    :param trajectory_file_name: The name of file with trajectory
    :param with_timestamps: Indicates that a trajectory with timestamps is given
    """
    rgb_images = __read_dir(path_to_dataset / color_dir)
    point_clouds = __read_dir(path_to_dataset / point_clouds_dir)

    traj = __load_poses(path_to_dataset / trajectory_file_name, with_timestamps)

    database = Database.from_point_clouds(rgb_images, point_clouds, traj)
    return database
