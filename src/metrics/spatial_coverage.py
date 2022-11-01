from src.core import Database, VoxelGrid
from src.metrics.reduction_metric import ReductionMetric


class SpatialCoverage(ReductionMetric):
    """
    The method returns the ratio of
    the number of points in the map from the reduced database after down sampling
    to the number of points in the map from the original database after down sampling
    """

    def __init__(self, voxel_size: float = 0.1):
        """
        Constructs WholeSceneCoverage reduction metric
        :param voxel_size: Voxel size for down sampling
        """
        self.voxel_size = voxel_size

    def evaluate(self, original_bd: Database, filtered_db: Database) -> float:
        min_bounds, max_bounds = original_bd.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)
        original_map = original_bd.build_sparse_map(voxel_grid)
        filtered_map = filtered_db.build_sparse_map(voxel_grid)
        return len(filtered_map.point.positions) / len(original_map.point.positions)

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
