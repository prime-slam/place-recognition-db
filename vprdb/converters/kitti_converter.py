import open3d as o3d
import pykitti
import numpy as np
import mrob


def kitti_to_lib_format(
    source_dir: str,
    sequence: str,
    camera: int,
    output_pcd_dir: str,
    output_poses_dir: str,
):
    if camera != 2 and camera != 3:
        raise ValueError("Invalid camera choice. Acceptable values are '2' or '3'.")

    dataset = pykitti.odometry(source_dir, sequence)

    if camera == 2:
        cam_to_velo = dataset.calib.T_cam2_velo
        cam_intrinsics = dataset.calib.K_cam2
    else:
        cam_to_velo = dataset.calib.T_cam3_velo
        cam_intrinsics = dataset.calib.K_cam3

    poses = dataset.poses
    poses_to_write = []

    with open(output_poses_dir + "/" + "poses.txt", "w") as file:
        for i in range(len(dataset)):
            pcd = dataset.get_velo(i)
            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(pcd[:, :3])

            t_pcd_obj = o3d.t.geometry.PointCloud.from_legacy(pcd_obj)
            transformed_pcd = t_pcd_obj.transform(cam_to_velo)

            if camera == 2:
                image = dataset.get_cam2(i)
            else:
                image = dataset.get_cam3(i)

            width, height = image.size
            depth_reproj = transformed_pcd.project_to_depth_image(
                width, height, cam_intrinsics, depth_max=5000.0
            )

            pcd_from_depth = o3d.t.geometry.PointCloud.create_from_depth_image(
                depth_reproj, cam_intrinsics, depth_max=10000.0
            )
            converted_pcd = o3d.t.geometry.PointCloud.to_legacy(pcd_from_depth)

            pcd_filename = "{:06d}.pcd".format(i)
            filepath = output_pcd_dir + "/" + pcd_filename
            o3d.io.write_point_cloud(filepath, converted_pcd)

            cam0_to_velo = dataset.calib.T_cam0_velo

            pose_new = poses[i] @ cam0_to_velo @ np.linalg.inv(cam_to_velo)

            tx, ty, tz = pose_new[:3, 3]
            R = pose_new[:3, :3]

            qx, qy, qz, qw = mrob.geometry.so3_to_quat(R)
            t_quat = str(tx) + " " + str(ty) + " " + str(tz) + " "
            r_quat = str(qx) + " " + str(qy) + " " + str(qz) + " " + str(qw)

            quat = t_quat + r_quat
            poses_to_write.append(quat)

            file.write(poses_to_write[i] + "\n")
