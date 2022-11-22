import os

from src.core import Database
from src.io import read_dataset
from tests.utils import generate_trajectory_from_positions

artificial_traj_1 = generate_trajectory_from_positions(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 1, 1], [4, 1, 1]]
)
artificial_traj_2 = generate_trajectory_from_positions(
    [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
)

path_to_db = os.path.join("tests", "test_db")
real_db = read_dataset(path_to_db)
artificial_db_1 = Database(artificial_traj_1, real_db.images, real_db.pcds)
artificial_db_2 = Database(artificial_traj_2, real_db.images, real_db.pcds)
