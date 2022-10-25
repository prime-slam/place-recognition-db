import copy

from nptyping import Float, NDArray, Shape
from src.core import Database, PointCloud
from src.metrics.reduction_metric import ReductionMetric
from tqdm.contrib import tzip


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
        bounds = original_db.get_bounds()
        result = self._evaluate_with_bounds(original_db, filtered_db, bounds)
        return result

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__

    def evaluate_many(
        self, original_db: Database, filtered_dbs: list[Database]
    ) -> list[int]:
        bounds = original_db.get_bounds()

        results = []
        for filtered_db in filtered_dbs:
            results.append(self._evaluate_with_bounds(original_db, filtered_db, bounds))

        return results

    evaluate_many.__doc__ = ReductionMetric.evaluate_many.__doc__

    def _evaluate_with_bounds(
        self,
        original_db: Database,
        filtered_db: Database,
        bounds: tuple[NDArray[Shape["3"], Float], NDArray[Shape["3"], Float]],
    ) -> int:
        filtered_db_map = filtered_db.build_sparse_map(bounds, self.voxel_size)
        result = 0
        for pose, pcd in tzip(original_db.trajectory, original_db.pcds):
            if pcd in filtered_db.pcds:
                continue
            pcd = pcd.read().transform(pose)
            pcd = PointCloud.voxel_down_sample(pcd, bounds, self.voxel_size)
            united_map = copy.deepcopy(filtered_db_map)
            united_map += pcd
            united_map = PointCloud.voxel_down_sample(
                united_map, bounds, self.voxel_size
            )
            difference = len(united_map.points) - len(filtered_db_map.points)
            if ((len(pcd.points) - difference) / len(pcd.points)) < self.threshold:
                result += 1

        return result
