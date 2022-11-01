import open3d as o3d

from src.core.voxel_grid import VoxelGrid


def voxel_down_sample(
    point_cloud: o3d.t.geometry.PointCloud,
    voxel_grid: VoxelGrid,
) -> o3d.t.geometry.PointCloud:
    """
    Voxel down sampling with voxel grid given
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
