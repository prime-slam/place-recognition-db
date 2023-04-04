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

from tests.test_data import artificial_db_1, artificial_db_2
from vprdb.core import Database
from vprdb.reduction_methods import EveryNth

res_traj_indices_1 = [0, 2, 4]
res_traj_indices_2 = [0, 4]
res_traj_indices_3 = [0, 2, 4]
res_traj_indices_4 = [0, 4]


@pytest.mark.parametrize(
    "input_db, n, res_traj_indices",
    [
        (artificial_db_1, 2, res_traj_indices_1),
        (artificial_db_1, 4, res_traj_indices_2),
        (artificial_db_2, 2, res_traj_indices_3),
        (artificial_db_2, 4, res_traj_indices_4),
    ],
)
def test_every_nth(input_db: Database, n: int, res_traj_indices: list):
    every_nth_method = EveryNth(n)
    new_db = every_nth_method.reduce(input_db)
    assert (
        np.asarray(input_db.trajectory)[res_traj_indices]
        == np.asarray(new_db.trajectory)
    ).all()
