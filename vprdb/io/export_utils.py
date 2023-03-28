#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import mrob
import numpy as np
import shutil

from pathlib import Path

from vprdb.core import Database


def __poses_to_txt(poses, filename_to_write):
    with open(filename_to_write, "w") as trajectory_file:
        for pose in poses:
            R = pose[:3, :3]
            t = pose[:3, 3]
            quat = mrob.geometry.so3_to_quat(R)
            pose_string = " ".join(np.concatenate((t, quat)).astype(str))
            trajectory_file.write(f"{pose_string}\n")


def export(
    database: Database,
    path_to_export: Path,
    color_dir: str,
    point_clouds_dir: str,
    trajectory_file_name: str,
):
    """
    Exports Database to hard drive
    :param database: Database for exporting
    :param path_to_export: Directory for exporting. Will be created if it does not exist
    :param color_dir: Directory name for saving color images
    :param point_clouds_dir: Directory name for saving depth images / PCDs
    :param trajectory_file_name: File name for saving the trajectory
    """
    path_to_color = path_to_export / color_dir
    path_to_point_clouds = path_to_export / point_clouds_dir
    path_to_color.mkdir(parents=True, exist_ok=False)
    path_to_point_clouds.mkdir(exist_ok=False)

    for rgb_image in database.color_images:
        shutil.copyfile(rgb_image.path, path_to_color / rgb_image.path.name)

    for point_cloud in database.point_clouds:
        shutil.copyfile(point_cloud.path, path_to_point_clouds / point_cloud.path.name)

    __poses_to_txt(database.trajectory, path_to_export / trajectory_file_name)
