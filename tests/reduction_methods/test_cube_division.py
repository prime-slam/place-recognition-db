import pytest

from src.core import Database
from src.reduction_methods import CubeDivision
from tests.test_data import artificial_db_1, artificial_db_2
from tests.utils import extract_positions_from_trajectory

res_positions_1 = [[1, 1, 0], [3, 1, 1]]
res_positions_2 = [[3, 1, 1]]
res_positions_3 = [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
res_positions_4 = [[10, 0, 0], [30, 10, 10], [40, 10, 10]]


@pytest.mark.parametrize(
    "input_db, cube_size, res_positions",
    [
        (artificial_db_1, 2.0, res_positions_1),
        (artificial_db_1, 4.0, res_positions_2),
        (artificial_db_2, 2.0, res_positions_3),
        (artificial_db_2, 16.0, res_positions_4),
    ],
)
def test_cube_division(input_db: Database, cube_size: float, res_positions: list):
    cube_division_method = CubeDivision(cube_size)
    new_db = cube_division_method.reduce(input_db)
    new_db_positions = extract_positions_from_trajectory(new_db.trajectory)
    assert new_db_positions == res_positions
