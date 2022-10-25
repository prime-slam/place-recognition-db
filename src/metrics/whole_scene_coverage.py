from src.core import Database
from src.metrics.reduction_metric import ReductionMetric


class WholeSceneCoverage(ReductionMetric):
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
        bounds = original_bd.get_bounds()
        original_map = original_bd.build_sparse_map(bounds, self.voxel_size)
        filtered_map = filtered_db.build_sparse_map(bounds, self.voxel_size)
        return len(filtered_map.points) / len(original_map.points)

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__

    def evaluate_many(
        self, original_bd: Database, filtered_dbs: list[Database]
    ) -> list[float]:
        bounds = original_bd.get_bounds()
        original_map = original_bd.build_sparse_map(bounds, self.voxel_size)
        results = []
        for filtered_db in filtered_dbs:
            filtered_map = filtered_db.build_sparse_map(bounds, self.voxel_size)
            results.append(len(filtered_map.points) / len(original_map.points))
        return results

    evaluate_many.__doc__ = ReductionMetric.evaluate_many.__doc__
