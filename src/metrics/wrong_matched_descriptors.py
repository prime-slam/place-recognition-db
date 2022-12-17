import faiss

from tqdm import tqdm

from src.core import Database, voxel_down_sample, VoxelGrid
from src.cos_place import CosPlace
from src.metrics.reduction_metric import ReductionMetric


class WrongMatchedDescriptors(ReductionMetric):
    """
    The metric returns number of frames that were mismatched by CosPlace.
    Checking is based on the intersection of two matched clouds.
    """

    def __init__(self, threshold: float, cos_place: CosPlace, voxel_size: float = 0.1):
        """
        Constructs WrongMatchedDescriptors reduction metric
        :param threshold: Percentage of frame coverage, below which the frame will be considered uncovered
        :param cos_place: CosPlace instance for calculating image descriptors
        :param voxel_size: Voxel size for down sampling
        """
        self.threshold = threshold
        self.cos_place = cos_place
        self.voxel_size = voxel_size

    def evaluate(self, original_db: Database, filtered_db: Database) -> int:
        filtered_db_images_set = set(filtered_db.images)
        queries_images = []
        queries_pcds = []
        queries_traj = []
        for i, image in enumerate(original_db.images):
            if image not in filtered_db_images_set:
                queries_images.append(image)
                queries_pcds.append(original_db.pcds[i])
                queries_traj.append(original_db.trajectory[i])
        queries_db = Database(queries_traj, queries_images, queries_pcds)
        queries_descriptors = self.cos_place.get_database_descriptors(queries_db)
        db_descriptors = self.cos_place.get_database_descriptors(filtered_db)

        faiss_index = faiss.IndexFlatL2(self.cos_place.fc_output_dim)
        faiss_index.add(db_descriptors)

        del db_descriptors

        _, predictions = faiss_index.search(queries_descriptors, 1)

        min_bounds, max_bounds = original_db.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)

        not_covered = 0
        for query_index, prediction in tqdm(
            enumerate(predictions), total=len(predictions)
        ):
            query_pcd = queries_db.get_pcd_by_index(query_index)
            query_pose = queries_db.trajectory[query_index]
            db_pcd = filtered_db.get_pcd_by_index(prediction[0])
            db_pose = filtered_db.trajectory[prediction[0]]

            query_pcd = query_pcd.transform(query_pose)
            db_pcd = db_pcd.transform(db_pose)

            query_pcd = voxel_down_sample(query_pcd, voxel_grid)
            db_pcd = voxel_down_sample(db_pcd, voxel_grid)

            united_map = voxel_down_sample(query_pcd + db_pcd, voxel_grid)
            difference = len(united_map.point.positions) - len(db_pcd.point.positions)
            query_size = len(query_pcd.point.positions)
            if ((query_size - difference) / query_size) <= self.threshold:
                not_covered += 1

        return not_covered

    evaluate.__doc__ = ReductionMetric.evaluate.__doc__
