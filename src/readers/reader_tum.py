import open3d as o3d
import mrob
import numpy as np
import os
import cv2

from reader import Reader
from nptyping import NDArray, Shape, Float, UInt8
from pathlib import Path
from src.database import Database


class ReaderTUM(Reader):
    def __init__(self, path_to_dataset: str, max_difference: int = 1000):
        self.path_to_dataset = path_to_dataset
        self.max_difference = max_difference

    def read_dataset(self) -> Database:
        return super().read_dataset()

    def _get_images_pcds_traj(
        self,
    ) -> tuple[list[str], list[str], list[NDArray[Shape["4, 4"], Float]]]:
        depth_with_timestamps = self.__read_folder(
            os.path.join(self.path_to_dataset, "depth")
        )
        rgb_with_timestamps = self.__read_folder(
            os.path.join(self.path_to_dataset, "rgb")
        )
        matches = self.__associate(rgb_with_timestamps, depth_with_timestamps)
        timestamps, traj = self.__read_trajectory()
        timestamps = np.asarray(timestamps)
        images_paths = []
        pcds_paths = []
        res_traj = []
        for (a, b) in matches:
            diff = abs(timestamps - np.mean([a, b]))
            min_diff_ind = np.argmin(diff)
            images_paths.append(rgb_with_timestamps[a])
            pcds_paths.append(depth_with_timestamps[b])
            res_traj.append(traj[min_diff_ind])
        return images_paths, pcds_paths, res_traj

    @staticmethod
    def _get_rgb(path_to_image: str) -> NDArray[Shape["*, *, 3"], UInt8]:
        return cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    @staticmethod
    def _get_pcd(path_to_pcd: str) -> o3d.geometry.PointCloud:
        depth_image = cv2.imread(path_to_pcd, cv2.IMREAD_ANYDEPTH)
        depth_image = o3d.geometry.Image(depth_image)
        intrinsic_matrix = np.asarray([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.intrinsic_matrix = intrinsic_matrix
        intrinsic.width, intrinsic.height = 640, 480
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, intrinsic, depth_scale=5000
        )
        return pcd

    def __read_trajectory(
        self,
    ) -> tuple[list[float], list[NDArray[Shape["4, 4"], Float]]]:
        poses_quat = []
        with open(os.path.join(self.path_to_dataset, "groundtruth.txt"), "r") as file:
            for line in file:
                if line[0] == "#":
                    continue
                poses_quat.append([float(i) for i in line.split(" ")])

        Ts = []
        timestamps = []
        for i, pose in enumerate(poses_quat):
            timestamps.append(pose[0])
            t = pose[1:4]
            R = mrob.geometry.quat_to_so3(pose[4:8])
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            Ts.append(T)
        return timestamps, Ts

    @staticmethod
    def __read_folder(path_to_folder: str) -> dict[float, str]:
        files = os.listdir(path_to_folder)
        timestamps = [float(Path(file).stem) for file in files]
        files = [os.path.join(path_to_folder, x) for x in files]
        timestamp_image_kvp = dict(zip(timestamps, files))
        return timestamp_image_kvp

    def __associate(
        self, rgb_with_timestamps: dict, depth_with_timestamps: dict
    ) -> list:
        color_keys = np.asarray(list(rgb_with_timestamps.keys()))
        depth_keys = np.asarray(list(depth_with_timestamps.keys()))
        best_matches = list()
        for timestamp in color_keys:
            best_match = depth_keys[np.argmin(np.abs(depth_keys - timestamp))]
            if abs(best_match - timestamp) < self.max_difference:
                best_matches.append((timestamp, best_match))
        return sorted(best_matches)
