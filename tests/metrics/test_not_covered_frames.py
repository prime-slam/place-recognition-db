import numpy as np
import pytest

from src.core import Database
from src.metrics import NotCoveredFrames
from src.reduction_methods import SetCover
from tests.test_data import real_db


filtered_dbs = []
for i in range(len(real_db)):
    set_cover_method = SetCover(i + 1)
    new_datab = set_cover_method.reduce(real_db)
    filtered_dbs.append(new_datab)


@pytest.mark.parametrize(
    "filtered_db",
    filtered_dbs,
)
def test_not_covered_frames_zero_threshold(filtered_db: Database):
    assert NotCoveredFrames(0).evaluate(real_db, filtered_db) == 0


thresholds = np.arange(0.1, 0.9, 0.1)


@pytest.mark.parametrize(
    "threshold",
    thresholds,
)
def test_not_covered_frames_ordering(threshold: float):
    """
    Checks that the number of uncovered frames increases
    as the number of frames in the new database decreases
    """
    metric_results = []
    for filtered_db in filtered_dbs:
        metric_result = NotCoveredFrames(threshold).evaluate(real_db, filtered_db)
        metric_results.append(metric_result)
    metric_results = np.asarray(metric_results[::-1])
    assert np.all(metric_results[:-1] <= metric_results[1:])
