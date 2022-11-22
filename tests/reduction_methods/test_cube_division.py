import numpy as np
import pytest

from src.core import Database
from src.reduction_methods import CubeDivision
from tests.test_data import artificial_db_1, artificial_db_2

res_traj_indices_1 = [2, 3]
res_traj_indices_2 = [3]
res_traj_indices_3 = [0, 1, 2, 3, 4]
res_traj_indices_4 = [1, 3, 4]


@pytest.mark.parametrize(
    "input_db, cube_size, res_traj_indices",
    [
        (artificial_db_1, 2.0, res_traj_indices_1),
        (artificial_db_1, 4.0, res_traj_indices_2),
        (artificial_db_2, 2.0, res_traj_indices_3),
        (artificial_db_2, 16.0, res_traj_indices_4),
    ],
)
def test_cube_division(input_db: Database, cube_size: float, res_traj_indices: list):
    cube_division_method = CubeDivision(cube_size)
    new_db = cube_division_method.reduce(input_db)
    assert (
        np.asarray(input_db.trajectory)[res_traj_indices]
        == np.asarray(new_db.trajectory)
    ).all()
