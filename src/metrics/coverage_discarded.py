import copy

from src.core import Database
from src.metrics.reduction_metric import ReductionMetric
from src.metrics.utils import build_map, get_bounds
from tqdm.contrib import tzip


class CoverageDiscarded(ReductionMetric):
    """
    The metric returns number of frames not covered by the database
    """

    def __init__(self, voxel_size: float, threshold: float):
        """
        Constructs CoverageDiscarded reduction metric
        :param voxel_size: Voxel size for down sampling
        :param threshold: Coverage threshold for the frame to be considered uncovered
        """
        self.voxel_size = voxel_size
        self.threshold = threshold

    def evaluate(self, original_db: Database, filtered_db: Database) -> int:
        min_bound, max_bound = get_bounds(original_db)

        filtered_db_map = build_map(filtered_db, min_bound, max_bound, self.voxel_size)
        result = 0
        for pose, pcd in tzip(original_db.trajectory, original_db.pcds):
            if pcd in filtered_db.pcds:
                continue
            pcd = pcd.read().transform(pose)
            pcd, _, _ = pcd.voxel_down_sample_and_trace(
                self.voxel_size, min_bound, max_bound
            )
            united_map = copy.deepcopy(filtered_db_map)
            united_map += pcd
            united_map, _, _ = united_map.voxel_down_sample_and_trace(
                self.voxel_size, min_bound, max_bound
            )
            difference = len(united_map.points) - len(filtered_db_map.points)
            if ((len(pcd.points) - difference) / len(pcd.points)) < self.threshold:
                result += 1

        return result

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
