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
import pytest

from tests.test_data import real_db
from tests.utils import generate_random_samples_of_test_db, get_db_subset
from vprdb.metrics import frames_coverage


@pytest.mark.parametrize("indices", generate_random_samples_of_test_db(10))
def test_frames_coverage_db_subset(indices):
    """
    The metric should return 1 for those frames that are both
    in the source database and in its subset
    """
    new_db = get_db_subset(real_db, indices)
    metric_result = frames_coverage(real_db, new_db)
    assert (np.asarray(metric_result)[indices] == 1).all()
