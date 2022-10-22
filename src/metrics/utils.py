import numpy as np
import open3d as o3d

from nptyping import Float, NDArray, Shape
from src.core import Database


def build_map(
    db: Database,
    min_bound: NDArray[Shape["3"], Float],
    max_bound: NDArray[Shape["3"], Float],
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    global_pcd = o3d.geometry.PointCloud()
    for pose, pcd, i in zip(db.trajectory, db.pcds, range(len(db))):
        pcd = pcd.read().transform(pose)
        global_pcd += pcd
        if i % 100 == 0:
            global_pcd, _, _ = global_pcd.voxel_down_sample_and_trace(
                voxel_size, min_bound, max_bound
            )
    global_pcd, _, _ = global_pcd.voxel_down_sample_and_trace(
        voxel_size, min_bound, max_bound
    )
    return global_pcd


def get_bounds(
    db: Database,
) -> tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]]:
    min_bounds = np.empty((0, 3))
    max_bounds = np.empty((0, 3))

    for pose, pcd in zip(db.trajectory, db.pcds):
        pcd = pcd.read().transform(pose)
        min_bounds = np.append(min_bounds, [pcd.get_min_bound()], axis=0)
        max_bounds = np.append(max_bounds, [pcd.get_max_bound()], axis=0)

    min_bound = np.amin(min_bounds, axis=0)
    max_bound = np.amax(max_bounds, axis=0)
    return min_bound, max_bound
