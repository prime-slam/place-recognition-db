import numpy as np

from nptyping import Float, NDArray, Shape


def generate_trajectory_from_positions(
    xyz_positions: list,
) -> list[NDArray[Shape["4, 4"], Float]]:
    poses = []
    for xyz in xyz_positions:
        pose = np.empty((4, 4))
        pose[:3, 3] = xyz
        poses.append(np.asarray(pose))
    return poses
