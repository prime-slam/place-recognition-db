from tqdm.contrib import tzip

from src.core import Database, voxel_down_sample, VoxelGrid
from src.metrics.reduction_metric import ReductionMetric


class NotCoveredFrames(ReductionMetric):
    """
    The metric returns number of frames not covered by the database
    """

    def __init__(self, threshold: float, voxel_size: float = 0.1):
        """
        Constructs CoverageDiscarded reduction metric
        :param voxel_size: Voxel size for down sampling
        :param threshold: Coverage threshold for the frame to be considered uncovered
        """
        self.threshold = threshold
        self.voxel_size = voxel_size

    def evaluate(self, original_db: Database, filtered_db: Database) -> int:
        min_bounds, max_bounds = original_db.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)
        filtered_db_map = filtered_db.build_sparse_map(voxel_grid)
        result = 0
        for pose, pcd_raw in tzip(original_db.trajectory, original_db.pcds):
            if pcd_raw in filtered_db.pcds:
                continue
            pcd = pcd_raw.point_cloud.transform(pose)
            pcd = voxel_down_sample(pcd, voxel_grid)
            united_map = filtered_db_map.clone()
            united_map += pcd
            united_map = voxel_down_sample(united_map, voxel_grid)
            difference = len(united_map.point.positions) - len(
                filtered_db_map.point.positions
            )
            pcd_length = len(pcd.point.positions)
            if ((pcd_length - difference) / pcd_length) < self.threshold:
                result += 1
        return result

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
