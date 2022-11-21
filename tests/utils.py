import mrob
import numpy as np

from nptyping import Float, NDArray, Shape
from src.core import Database
from src.reduction_methods import ReductionMethod


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


def generate_traj(xyz_list: list):
    poses = []
    for xyz in xyz_list:
        pose = np.empty((4, 4))
        pose[:3, 3] = xyz
        poses.append(np.asarray(pose))
    return poses


def get_new_traj(reduction_method: ReductionMethod, input_db: Database):
    new_db = reduction_method.reduce(input_db)
    new_db_traj = [pose[:3, 3].tolist() for pose in new_db.trajectory]
    return new_db_traj
