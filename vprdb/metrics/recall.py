import numpy as np

from vprdb.core import calculate_point_cloud_coverage, Database, VoxelGrid


def recall(
    source_db: Database,
    test_db: Database,
    matches: list[int],
    voxel_size: float = 0.3,
    threshold: float = 0.3,
) -> float:
    """
    The metric finds the number of correctly matched frames

    :param source_db: Database for the VPR task
    :param test_db: Database used as queries to the VPR system
    :param matches: VPR system matches
    :param voxel_size: Voxel size for down sampling
    :param threshold: The value of frame coverage,
    below which the frame will be considered uncovered

    :return: Recall value
    """
    if len(test_db) != len(matches):
        raise ValueError(
            "The length of the matches and the test database must be the same"
        )

    min_bounds_test, max_bounds_test = test_db.bounds
    min_bounds_source, max_bounds_source = source_db.bounds
    min_bounds = np.amin(np.row_stack((min_bounds_test, min_bounds_source)), axis=0)
    max_bounds = np.amax(np.row_stack((max_bounds_test, max_bounds_source)), axis=0)

    voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)
    results = []
    for i, match in enumerate(matches):
        pose_query = test_db.trajectory[i]
        pcd_query = test_db.point_clouds[i].point_cloud.transform(pose_query)

        pose_source = source_db.trajectory[match]
        pcd_source = source_db.point_clouds[match].point_cloud.transform(pose_source)

        coverage = calculate_point_cloud_coverage(pcd_query, pcd_source, voxel_grid)
        results.append(coverage > threshold)

    return sum(results) / len(results)
