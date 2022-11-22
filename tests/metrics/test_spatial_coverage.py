import numpy as np

from src.metrics import SpatialCoverage
from src.reduction_methods import SetCover
from tests.test_data import real_db

metric_results = []
for i in range(len(real_db)):
    set_cover_method = SetCover(i + 1)
    new_datab = set_cover_method.reduce(real_db)
    metric_result = SpatialCoverage().evaluate(real_db, new_datab)
    metric_results.append(metric_result)
metric_results = np.asarray(metric_results)


def test_spatial_coverage_ordering():
    """
    Checks that the result of the metric decreases
    with a decrease in the number of frames in the filtered database
    :return:
    """
    assert np.all(metric_results[:-1] < metric_results[1:])


def test_spatial_coverage_full():
    assert metric_results[-1] == 1
