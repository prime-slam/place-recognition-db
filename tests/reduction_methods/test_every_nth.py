import numpy as np
import pytest

from src.core import Database
from src.reduction_methods import EveryNth
from tests.test_data import artificial_db_1, artificial_db_2

res_traj_indices_1 = [0, 2, 4]
res_traj_indices_2 = [0, 4]
res_traj_indices_3 = [0, 2, 4]
res_traj_indices_4 = [0, 4]


@pytest.mark.parametrize(
    "input_db, n, res_traj_indices",
    [
        (artificial_db_1, 2, res_traj_indices_1),
        (artificial_db_1, 4, res_traj_indices_2),
        (artificial_db_2, 2, res_traj_indices_3),
        (artificial_db_2, 4, res_traj_indices_4),
    ],
)
def test_every_nth(input_db: Database, n: int, res_traj_indices: list):
    every_nth_method = EveryNth(n)
    new_db = every_nth_method.reduce(input_db)
    assert (
        np.asarray(input_db.trajectory)[res_traj_indices]
        == np.asarray(new_db.trajectory)
    ).all()
