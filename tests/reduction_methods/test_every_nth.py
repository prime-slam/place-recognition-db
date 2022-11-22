import pytest

from src.core import Database
from src.reduction_methods import EveryNth
from tests.test_data import artificial_db_1, artificial_db_2
from tests.utils import extract_positions_from_trajectory

res_positions_1 = [[0, 0, 0], [1, 1, 0], [4, 1, 1]]
res_positions_2 = [[0, 0, 0], [4, 1, 1]]
res_positions_3 = [[0, 0, 0], [15, 15, 0], [40, 10, 10]]
res_positions_4 = [[0, 0, 0], [40, 10, 10]]


@pytest.mark.parametrize(
    "input_db, n, res_positions",
    [
        (artificial_db_1, 2, res_positions_1),
        (artificial_db_1, 4, res_positions_2),
        (artificial_db_2, 2, res_positions_3),
        (artificial_db_2, 4, res_positions_4),
    ],
)
def test_every_nth(input_db: Database, n: int, res_positions: list):
    every_nth_method = EveryNth(n)
    new_db = every_nth_method.reduce(input_db)
    new_db_positions = extract_positions_from_trajectory(new_db.trajectory)
    assert new_db_positions == res_positions
