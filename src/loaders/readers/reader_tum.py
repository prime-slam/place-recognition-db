import cv2
import mrob
import numpy as np
import open3d as o3d
import os

from nptyping import Float, NDArray, Shape, UInt8
from pathlib import Path

from src.loaders.providers import ImageProvider, PointCloudProvider
from src.loaders.readers.reader import Reader


class ReaderTUM(Reader):
    def __init__(
        self,
        path_to_dataset: str,
        intrinsics: NDArray[Shape["3, 3"], Float],
        dist_coeff: NDArray[Shape["5"], Float],
        max_difference: int = 1000,
    ):
        self.path_to_dataset = path_to_dataset
        self.max_difference = max_difference
        self.intrinsics = intrinsics
        self.dist_coeff = dist_coeff
        self._depth_scale = 5000
        self._depth_max = 10

    def _get_images_pcds_traj(
        self,
    ) -> tuple[
        list[ImageProvider],
        list[PointCloudProvider],
        list[NDArray[Shape["4, 4"], Float]],
    ]:
        depth_with_timestamps = self.__read_folder(
            os.path.join(self.path_to_dataset, "depth")
        )
        rgb_with_timestamps = self.__read_folder(
            os.path.join(self.path_to_dataset, "rgb")
        )
        matches = self.__associate_depth_with_rgb(
            rgb_with_timestamps, depth_with_timestamps
        )
        traj_timestamps, traj = self.__read_trajectory()
        images = []
        pcds = []
        res_traj = []
        for (rgb_timestamp, depth_timestamp) in matches:
            associated_pose_index = self.__associate_depth_and_rgb_with_traj(
                traj_timestamps, rgb_timestamp, depth_timestamp
            )
            res_traj.append(traj[associated_pose_index])
            images.append(
                ImageProvider(rgb_with_timestamps[rgb_timestamp], self._get_rgb)
            )
            pcds.append(
                PointCloudProvider(
                    depth_with_timestamps[depth_timestamp], self._get_pcd
                )
            )
        return images, pcds, res_traj

    def _get_rgb(self, path_to_image: str) -> NDArray[Shape["*, *, 3"], UInt8]:
        image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
        color_undistorted, new_color_intrinsics = self.__undistort(image)
        return color_undistorted

    def _get_pcd(self, path_to_pcd: str) -> o3d.t.geometry.PointCloud:
        depth_image = cv2.imread(path_to_pcd, cv2.IMREAD_ANYDEPTH)
        depth_undistorted, new_depth_intrinsics = self.__undistort(depth_image)
        depth_image = o3d.t.geometry.Image(depth_undistorted)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_image,
            new_depth_intrinsics,
            depth_scale=self._depth_scale,
            depth_max=self._depth_max,
        )
        return pcd

    def __undistort(self, image):
        shape = image.shape[:2][::-1]
        undist_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
            self.intrinsics, self.dist_coeff, shape, 1, shape
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            self.intrinsics,
            self.dist_coeff,
            None,
            undist_intrinsics,
            shape,
            cv2.CV_32FC1,
        )
        undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)
        return undistorted, undist_intrinsics

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
        timestamp_image_key_value_pairs = dict(zip(timestamps, files))
        return timestamp_image_key_value_pairs

    def __associate_depth_with_rgb(
        self, rgb_with_timestamps: dict, depth_with_timestamps: dict
    ) -> list:
        color_keys = np.asarray(list(rgb_with_timestamps.keys()))
        depth_keys = np.asarray(list(depth_with_timestamps.keys()))
        best_matches = list()
        for timestamp in depth_keys:
            best_match = color_keys[np.argmin(np.abs(color_keys - timestamp))]
            if abs(best_match - timestamp) < self.max_difference:
                best_matches.append((best_match, timestamp))
        return sorted(best_matches)

    @staticmethod
    def __associate_depth_and_rgb_with_traj(
        traj_timestamps: list[float], rgb_timestamp: float, depth_timestamp: float
    ):
        diff = abs(
            np.asarray(traj_timestamps) - np.mean([rgb_timestamp, depth_timestamp])
        )
        return np.argmin(diff)
