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
from vprdb.reduction_methods import CubeDivision

res_traj_indices_1 = [2, 3, 4]
res_traj_indices_2 = [3, 4]
res_traj_indices_3 = [0, 1, 2, 3, 4]
res_traj_indices_4 = [1, 3, 4]


@pytest.mark.parametrize(
    "input_db, cube_size, res_traj_indices",
    [
        (artificial_db_1, 2.0, res_traj_indices_1),
        (artificial_db_1, 4.0, res_traj_indices_2),
        (artificial_db_2, 2.0, res_traj_indices_3),
        (artificial_db_2, 16.0, res_traj_indices_4),
    ],
)
def test_cube_division(input_db: Database, cube_size: float, res_traj_indices: list):
    cube_division_method = CubeDivision(cube_size)
    new_db = cube_division_method.reduce(input_db)
    assert (
        np.asarray(input_db.trajectory)[res_traj_indices]
        == np.asarray(new_db.trajectory)
    ).all()
