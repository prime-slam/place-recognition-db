import os

from joblib import Parallel, delayed

from src.core import calculate_point_cloud_coverage, Database, VoxelGrid
from src.metrics.reduction_metric import ReductionMetric


class NotCoveredFrames(ReductionMetric):
    """
    The metric returns number of frames not covered by the database
    """

    def __init__(self, threshold: float, voxel_size: float = 0.1):
        """
        Constructs CoverageDiscarded reduction metric
        :param voxel_size: Voxel size for down sampling
        :param threshold: Percentage of frame coverage, below which the frame will be considered uncovered
        """
        self.threshold = threshold
        self.voxel_size = voxel_size

    def evaluate(self, original_db: Database, filtered_db: Database) -> int:
        min_bounds, max_bounds = original_db.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)
        filtered_db_map_pcds_set = set(filtered_db.pcds)

        def is_not_covered(pose, pcd_raw):
            if pcd_raw in filtered_db_map_pcds_set:
                return False
            for i, filtered_db_pcd in enumerate(filtered_db.pcds):
                pcd_query = pcd_raw.point_cloud.transform(pose)
                pcd_db = filtered_db_pcd.point_cloud.transform(
                    filtered_db.trajectory[i]
                )

                coverage = calculate_point_cloud_coverage(pcd_query, pcd_db, voxel_grid)
                if coverage >= self.threshold:
                    return False
            return True

        coverage_results = Parallel(n_jobs=os.cpu_count())(
            delayed(is_not_covered)(pose, pcd_raw)
            for pose, pcd_raw in zip(original_db.trajectory, original_db.pcds)
        )
        return sum(coverage_results)

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
