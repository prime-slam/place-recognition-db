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
import cv2
import open3d as o3d

from dataclasses import dataclass
from nptyping import Float, NDArray, Shape
from pathlib import Path


@dataclass(frozen=True)
class DepthImageProvider:
    """ Depth image provider is a wrapper for depth images """
    path: Path
    """ Path to file on hard drive """
    intrinsics: NDArray[Shape["3, 3"], Float]
    """ Intrinsic camera parameters """
    depth_scale: int
    """ Depth scale for transforming depth """

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
        """ Returns Open3D point cloud """
        depth_image = cv2.imread(str(self.path), cv2.IMREAD_ANYDEPTH)
        height, width = depth_image.shape
        depth_image = o3d.geometry.Image(depth_image)
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, self.intrinsics)
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            intrinsics,
            depth_scale=self.depth_scale,
            depth_trunc=float("inf"),
        )
        return point_cloud
