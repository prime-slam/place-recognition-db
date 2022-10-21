import open3d as o3d
import psutil

from src.core import Database
from src.metrics.reduction_metric import ReductionMetric
from tqdm import tqdm

MEMORY_LIMIT = 70


class WholeSceneCoverage(ReductionMetric):
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size

    def evaluate(self, original_db: Database, filtered_db: Database) -> float:
        original_db_map = self.__build_scene_map(original_db)
        filtered_db_map = self.__build_scene_map(filtered_db)
        return len(filtered_db_map.points) / len(original_db_map.points)

    def __build_scene_map(self, db: Database) -> o3d.geometry.PointCloud:
        pcd_global = o3d.geometry.PointCloud()
        for i, pose in enumerate(tqdm(db.trajectory)):
            memory_usage_percentage = psutil.virtual_memory()[2]
            if memory_usage_percentage > MEMORY_LIMIT:
                raise MemoryError(
                    "The method requires too much RAM, try increasing the voxel size"
                )
            pcd = db.get_pcd_by_index(i).transform(pose)
            pcd_global += pcd
            memory_usage_percentage = psutil.virtual_memory()[2]
            if memory_usage_percentage > MEMORY_LIMIT:
                pcd_global = pcd_global.voxel_down_sample(self.voxel_size)
        pcd_global = pcd_global.voxel_down_sample(self.voxel_size)
        return pcd_global
