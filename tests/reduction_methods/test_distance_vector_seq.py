import pytest

from src.core import Database
from src.reduction_methods import DistanceVectorSeq
from tests.test_data import artificial_db_1, artificial_db_2
from tests.utils import extract_positions_from_trajectory

res_positions_1 = [[0, 0, 0], [3, 1, 1]]
res_positions_2 = [[0, 0, 0], [3, 1, 1]]
res_positions_3 = [[0, 0, 0], [10, 0, 0], [15, 15, 0], [30, 10, 10], [40, 10, 10]]
res_positions_4 = [[0, 0, 0], [15, 15, 0], [30, 10, 10]]


@pytest.mark.parametrize(
    "input_db, distance_threshold, res_positions",
    [
        (artificial_db_1, 1.5, res_positions_1),
        (artificial_db_1, 3.0, res_positions_2),
        (artificial_db_2, 1.5, res_positions_3),
        (artificial_db_2, 16.0, res_positions_4),
    ],
)
def test_distance_vector_seq(
    input_db: Database, distance_threshold: float, res_positions: list
):
    distance_vector_seq_method = DistanceVectorSeq(distance_threshold)
    new_db = distance_vector_seq_method.reduce(input_db)
    new_db_positions = extract_positions_from_trajectory(new_db.trajectory)
    assert new_db_positions == res_positions
