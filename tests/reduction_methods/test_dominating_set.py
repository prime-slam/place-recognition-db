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
from vprdb.core import Database
from vprdb.reduction_methods import DominatingSet


@pytest.mark.parametrize(
    "input_db, threshold, result_db_len",
    [
        (real_db, 0.05, 3),
        (real_db, 0.5, 4),
        (real_db, 0.8, 5),
    ],
)
def test_dominating_set(input_db: Database, threshold: float, result_db_len: int):
    """
    In the test database, there are two weakly overlapping frames, as well as two frames with great overlap.
    Therefore, a low threshold value should produce three frames in the database,
    a medium threshold value should result in four frames,
    and a high threshold value should generate a database with five frames.
    """
    dominating_set_method = DominatingSet(threshold=threshold)
    new_db = dominating_set_method.reduce(input_db)
    assert len(new_db) == result_db_len
