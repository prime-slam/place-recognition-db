import pytest

from src.core import Database
from src.reduction_methods import CubeDivision
from tests.test_data import artificial_db_1, artificial_db_2
from tests.utils import get_new_traj

res_traj_1 = [[1, 1, 0], [3, 1, 1]]
res_traj_2 = [[3, 1, 1]]
res_traj_3 = [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
res_traj_4 = [[10, 0, 0], [30, 10, 10], [40, 10, 10]]


@pytest.mark.parametrize(
    "input_db, cube_size, res_traj",
    [
        (artificial_db_1, 2.0, res_traj_1),
        (artificial_db_1, 4.0, res_traj_2),
        (artificial_db_2, 2.0, res_traj_3),
        (artificial_db_2, 16.0, res_traj_4),
    ],
)
def test_cube_division(input_db: Database, cube_size: float, res_traj: list):
    cube_division_method = CubeDivision(cube_size)
    new_db_traj = get_new_traj(cube_division_method, input_db)
    assert new_db_traj == res_traj
