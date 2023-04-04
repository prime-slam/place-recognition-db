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
import pytest

from tests.test_data import real_db
from tests.utils import filter_db, generate_random_samples_of_test_db
from vprdb.core import match_two_databases, VoxelGrid
from vprdb.metrics import recall


@pytest.mark.parametrize("indices", generate_random_samples_of_test_db())
def test_recall(indices):
    """
    Because the new database is being tested on the same database as it was generated,
    all frames in it can be correctly mapped
    """
    new_db = filter_db(real_db, indices)
    min_bounds, max_bounds = real_db.bounds
    voxel_grid = VoxelGrid(min_bounds, max_bounds, 0.3)
    matches = match_two_databases(real_db, new_db, voxel_grid)
    metric_result = recall(new_db, real_db, matches)
    assert metric_result >= (len(indices) / len(real_db))


def test_recall_best_case():
    metric_result = recall(real_db, real_db, list(range(len(real_db))))
    assert metric_result == 1
