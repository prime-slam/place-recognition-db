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
"""
Metrics allow to evaluate the quality of created databases,
as well as to estimate the accuracy of various VPR systems.
"""
from vprdb.metrics.frames_coverage_ import frames_coverage
from vprdb.metrics.recall_ import recall
from vprdb.metrics.spatial_coverage_ import spatial_coverage

__all__ = ["frames_coverage", "recall", "spatial_coverage"]
