import numpy as np

from nptyping import Float, NDArray, Shape
from tqdm import tqdm

from src.core import Database, memory, voxel_down_sample, VoxelGrid
from src.loaders.providers import PointCloudProvider
from src.reduction_methods.reduction_method import ReductionMethod


@memory.cache
def _get_voxel_by_frames_coverage_matrix(
    voxel_grid: VoxelGrid,
    pcds: list[PointCloudProvider],
    poses: list[NDArray[Shape["4, 4"], Float]],
):
    voxels_enum = dict()
    cur_voxel_num = 0
    frames_voxels = []
    for pcd_raw, pose in zip(pcds, poses):
        pcd = pcd_raw.point_cloud.transform(pose)
        down_sampled = voxel_down_sample(pcd, voxel_grid)
        frame_voxels = []
        for point in down_sampled.point.positions:
            voxel_index = voxel_grid.get_voxel_index(point.numpy())
            if voxel_index in voxels_enum:
                num = voxels_enum[voxel_index]
                frame_voxels.append(num)
            else:
                voxels_enum[voxel_index] = cur_voxel_num
                frame_voxels.append(cur_voxel_num)
                cur_voxel_num += 1
        frames_voxels.append(frame_voxels)

    coverage_mat = np.zeros((len(frames_voxels), len(voxels_enum))).astype(np.uint8)

    for i, voxels in enumerate(frames_voxels):
        coverage_mat[i, voxels] = 1

    return coverage_mat


class SetCover(ReductionMethod):
    """The method reduces database by set cover greedy algorithm"""

    def __init__(self, db_size: int, voxel_size: float = 0.1):
        """
        Constructs SetCover reduction method
        :param db_size: The number of frames that should remain after compression
        :param voxel_size: Voxel size for down sampling
        """
        self.db_size = db_size
        self.voxel_size = voxel_size

    def reduce(self, db: Database) -> Database:
        min_bounds, max_bounds = db.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)
        coverage_mat = _get_voxel_by_frames_coverage_matrix(
            voxel_grid, db.pcds, db.trajectory
        )

        chosen_frames = []
        for _ in tqdm(range(self.db_size)):
            chosen_frame = np.argmax(np.sum(coverage_mat, axis=1))
            coverage_mat = coverage_mat[:, coverage_mat[chosen_frame] == 0]
            chosen_frames.append(chosen_frame)

        chosen_frames.sort()
        new_traj = [db.trajectory[i] for i in chosen_frames]
        new_rgb = [db.images[i] for i in chosen_frames]
        new_pcds = [db.pcds[i] for i in chosen_frames]
        return Database(new_traj, new_rgb, new_pcds)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
