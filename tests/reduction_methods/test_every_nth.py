import pytest

from src.core import Database
from src.reduction_methods import EveryNth
from tests.test_data import artificial_db_1, artificial_db_2
from tests.utils import get_new_traj

res_traj_1 = [[0, 0, 0], [1, 1, 0], [4, 1, 1]]
res_traj_2 = [[0, 0, 0], [4, 1, 1]]
res_traj_3 = [[0, 0, 0], [15, 15, 0], [40, 10, 10]]
res_traj_4 = [[0, 0, 0], [40, 10, 10]]


@pytest.mark.parametrize(
    "input_db, n, res_traj",
    [
        (artificial_db_1, 2, res_traj_1),
        (artificial_db_1, 4, res_traj_2),
        (artificial_db_2, 2, res_traj_3),
        (artificial_db_2, 4, res_traj_4),
    ],
)
def test_distance_vector(input_db: Database, n: int, res_traj: list):
    cube_division_method = EveryNth(n)
    new_db_traj = get_new_traj(cube_division_method, input_db)
    assert new_db_traj == res_traj
