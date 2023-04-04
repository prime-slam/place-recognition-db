#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import numpy as np

from tests.test_data import real_db
from vprdb.metrics import spatial_coverage
from vprdb.reduction_methods import DominatingSet

thresholds = [0.05, 0.5, 0.8]
metric_results = []
for threshold in thresholds:
    dominating_set_method = DominatingSet(threshold)
    reduced_db = dominating_set_method.reduce(real_db)
    metric_result = spatial_coverage(real_db, reduced_db)
    metric_results.append(metric_result)
metric_results = np.asarray(metric_results)


def test_spatial_coverage_ordering():
    """
    Checks that the result of the metric decreases
    with a decrease in the number of frames in the filtered database
    """
    assert np.all(metric_results[:-1] < metric_results[1:])


def test_spatial_coverage_full():
    assert metric_results[-1] == 1
