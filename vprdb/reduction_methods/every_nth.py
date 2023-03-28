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
from vprdb.core import Database
from vprdb.reduction_methods.reduction_method import ReductionMethod


class EveryNth(ReductionMethod):
    """The method returns every N-th item from DB"""

    def __init__(self, n: int):
        """
        Constructs EveryNth reduction method
        :param n: Step of method
        """
        self.n = n

    def reduce(self, db: Database) -> Database:
        new_traj = db.trajectory[:: self.n]
        new_color = db.color_images[:: self.n]
        new_point_clouds = db.point_clouds[:: self.n]
        return Database(new_color, new_point_clouds, new_traj)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
