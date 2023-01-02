import open3d as o3d

from src.core.voxel_grid import VoxelGrid


def voxel_down_sample(
    point_cloud: o3d.t.geometry.PointCloud,
    voxel_grid: VoxelGrid,
) -> o3d.t.geometry.PointCloud:
    """
    Voxel down sampling with voxel grid given

    This method is implemented in the legacy point cloud,
    but is still missing in the tensor point cloud in Open3D 0.16.0
    :param point_cloud: Point cloud for down sampling
    :param voxel_grid: Voxel grid for down sampling
    :return: Down sampled point cloud
    """
    legacy_pcd = point_cloud.to_legacy()
    voxel_down_result, _, _ = legacy_pcd.voxel_down_sample_and_trace(
        voxel_grid.voxel_size, voxel_grid.min_bounds, voxel_grid.max_bounds
    )
    tensor_pcd = o3d.t.geometry.PointCloud.from_legacy(voxel_down_result)
    return tensor_pcd


def calculate_point_cloud_coverage(
    query_pcd: o3d.t.geometry.PointCloud,
    db_pcd: o3d.t.geometry.PointCloud,
    voxel_grid: VoxelGrid,
) -> float:
    """
    Calculates coverage for query point cloud by database point cloud
    :param query_pcd: query point cloud
    :param db_pcd: database point cloud
    :param voxel_grid: voxel grid for down sampling
    :return: coverage ∈ [0; 1]
    """
    query_pcd = voxel_down_sample(query_pcd, voxel_grid)
    db_pcd = voxel_down_sample(db_pcd, voxel_grid)

    united_map = voxel_down_sample(query_pcd + db_pcd, voxel_grid)
    difference = len(united_map.point.positions) - len(db_pcd.point.positions)
    query_size = len(query_pcd.point.positions)
    return (query_size - difference) / query_size
