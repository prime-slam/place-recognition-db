import os

from src.core import Database
from src.providers import ColorImageProvider, PointCloudProvider
from tests.utils import generate_traj, read_trajectory

path_to_db = os.path.join("tests", "test_db")
path_to_pcds = os.path.join(path_to_db, "pcds")
path_to_rgbs = os.path.join(path_to_db, "rgb")
pcds = [
    PointCloudProvider(os.path.join(path_to_pcds, pcd_file_name))
    for pcd_file_name in sorted(os.listdir(path_to_pcds))
]
rgbs = [
    ColorImageProvider(os.path.join(path_to_rgbs, rgb_file_name))
    for rgb_file_name in sorted(os.listdir(path_to_rgbs))
]
real_traj = read_trajectory(os.path.join(path_to_db, "groundtruth.txt"))
artificial_traj_1 = generate_traj(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 1, 1], [4, 1, 1]]
)
artificial_traj_2 = generate_traj(
    [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
)

real_db = Database(real_traj, rgbs, pcds)
artificial_db_1 = Database(artificial_traj_1, rgbs, pcds)
artificial_db_2 = Database(artificial_traj_2, rgbs, pcds)
