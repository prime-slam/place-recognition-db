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
from abc import ABC, abstractmethod

from vprdb.core import Database


class ReductionMethod(ABC):
    """
    A class to represent Database reduction method
    """

    @abstractmethod
    def reduce(self, db: Database) -> Database:
        """
        Method for reducing the Database
        :param db: Database for reduction
        :return: Reduced database
        """
        pass
