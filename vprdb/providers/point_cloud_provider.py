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
import open3d as o3d

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PointCloudProvider:
    """PointCloudProvider provider is a wrapper for point clouds"""

    path: Path
    """Path to the file on the hard drive"""

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
        """Returns Open3D point cloud"""
        return o3d.io.read_point_cloud(str(self.path))
