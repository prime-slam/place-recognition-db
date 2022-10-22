from src.core import Database
from src.metrics.reduction_metric import ReductionMetric
from src.metrics.utils import build_map, get_bounds


class WholeSceneCoverage(ReductionMetric):
    """
    The method returns the ratio of
    the number of points in the map from the reduced database after down sampling
    to the number of points in the map from the original database after down sampling
    """

    def __init__(self, voxel_size: float):
        """
        Constructs WholeSceneCoverage reduction metric
        :param voxel_size: Voxel size for down sampling
        """
        self.voxel_size = voxel_size

    def evaluate(self, original_db: Database, filtered_db: Database) -> float:
        min_bound, max_bound = get_bounds(original_db)
        original_db_map = build_map(original_db, min_bound, max_bound, self.voxel_size)
        filtered_db_map = build_map(filtered_db, min_bound, max_bound, self.voxel_size)
        return len(filtered_db_map.points) / len(original_db_map.points)

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
