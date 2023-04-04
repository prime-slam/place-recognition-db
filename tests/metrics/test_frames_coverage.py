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
from tests.utils import filter_db, generate_random_samples_of_test_db
from vprdb.metrics import frames_coverage


@pytest.mark.parametrize("indices", generate_random_samples_of_test_db())
def test_frames_coverage(indices):
    """The metric should return 1 for those frames that are left in the database"""
    new_db = filter_db(real_db, indices)
    metric_result = frames_coverage(real_db, new_db)
    assert (np.asarray(metric_result)[indices] == 1).all()
