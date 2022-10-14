import cv2
import numpy as np
import open3d as o3d
import os

from src.core.database import Database


class Exporter:
    def __init__(
        self,
        export_folder: str = "exported_data",
        trajectory_file_name: str = "trajectory",
        images_folder_name: str = "rgb",
        pcds_folder_name: str = "pcds",
    ):
        self.export_folder = export_folder
        self.trajectory_file_name = trajectory_file_name
        self.images_folder_name = images_folder_name
        self.pcds_folder_name = pcds_folder_name

    def __create_export_infrastructure(self, path_to_save: str) -> tuple[str, str, str]:
        exported_folder_path = os.path.join(path_to_save, self.export_folder)
        if os.path.isdir(exported_folder_path):
            # TODO: change exception type
            raise Exception("exported_data directory exists")
        os.mkdir(exported_folder_path)
        path_to_traj = os.path.join(
            exported_folder_path, f"{self.trajectory_file_name}.txt"
        )
        path_to_rgb = os.path.join(exported_folder_path, self.images_folder_name)
        path_to_pcds = os.path.join(exported_folder_path, self.pcds_folder_name)
        os.mkdir(path_to_rgb)
        os.mkdir(path_to_pcds)
        return path_to_traj, path_to_rgb, path_to_pcds

    def export(self, path_to_save: str, db: Database):
        path_to_traj, path_to_rgb, path_to_pcds = self.__create_export_infrastructure(
            path_to_save
        )
        with open(path_to_traj, "w") as traj:
            for pose in db.trajectory:
                R = pose[:3, :3].astype(np.str)
                t = pose[:3, 3].astype(np.str)
                flatten_pose = np.concatenate((R.flatten(), t))
                pose_string = " ".join(flatten_pose)
                traj.write(f"{pose_string}\n")
        for i in range(len(db)):
            image = db.get_rgb_image_by_index(i)
            cv2.imwrite(os.path.join(path_to_rgb, f"{i}.png"), image)
            pcd = db.get_pcd_by_index(i)
            o3d.io.write_point_cloud(os.path.join(path_to_pcds, f"{i}.pcd"), pcd)
