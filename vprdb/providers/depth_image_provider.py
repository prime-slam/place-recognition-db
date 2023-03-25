import cv2
import open3d as o3d

from dataclasses import dataclass
from nptyping import Float, NDArray, Shape
from pathlib import Path


@dataclass(frozen=True)
class DepthImageProvider:
    path: Path
    intrinsics: NDArray[Shape["3, 3"], Float]
    depth_scale: int

    @property
    def point_cloud(self) -> o3d.geometry.PointCloud:
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
