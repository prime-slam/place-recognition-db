import pytest

from src.reduction_methods import SetCover
from tests.test_data import real_db

resulting_amount_of_frames_params = [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "resulting_amount_of_frames", resulting_amount_of_frames_params
)
def test_set_cover(resulting_amount_of_frames: int):
    set_cover_method = SetCover(resulting_amount_of_frames)
    new_db = set_cover_method.reduce(real_db)
    assert len(new_db) == resulting_amount_of_frames
