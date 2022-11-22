import numpy as np
import pytest

from src.core import Database
from src.reduction_methods import DistanceVector
from tests.test_data import artificial_db_1, artificial_db_2

res_traj_indices_1 = [0, 2, 3]
res_traj_indices_2 = [0, 3]
res_traj_indices_3 = [0, 1, 2, 3, 4]
res_traj_indices_4 = [0, 2, 3]


@pytest.mark.parametrize(
    "input_db, distance_threshold, res_traj_indices",
    [
        (artificial_db_1, 1.5, res_traj_indices_1),
        (artificial_db_1, 3.0, res_traj_indices_2),
        (artificial_db_2, 1.5, res_traj_indices_3),
        (artificial_db_2, 16.0, res_traj_indices_4),
    ],
)
def test_distance_vector(
    input_db: Database, distance_threshold: float, res_traj_indices: list
):
    distance_vector_method = DistanceVector(distance_threshold)
    new_db = distance_vector_method.reduce(input_db)
    assert (
        np.asarray(input_db.trajectory)[res_traj_indices]
        == np.asarray(new_db.trajectory)
    ).all()
