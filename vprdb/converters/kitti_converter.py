import argparse
import mrob
import numpy as np
import open3d as o3d
import pykitti
import shutil

from pathlib import Path


def copy_img_folder(src_folder, out_folder, shape):
    out_folder.mkdir()
    for i in range(shape):
        img_filename = "{:06d}.png".format(i)
        source = src_folder / img_filename
        destination = out_folder / img_filename
        shutil.copyfile(source, destination)


def kitti_to_lib_format(args):
    if args.camera != "2" and args.camera != "3":
        raise ValueError("Invalid camera choice. Acceptable values are '2' or '3'.")

    dataset = pykitti.odometry(args.source_dir, args.sequence)
    output_dir = Path(args.output_dir)
    source_dir = Path(args.source_dir)

    if args.camera == "2":
        cam_to_velo = dataset.calib.T_cam2_velo
        cam_intrinsics = dataset.calib.K_cam2
        img_folder = "image_2"
    else:
        cam_to_velo = dataset.calib.T_cam3_velo
        cam_intrinsics = dataset.calib.K_cam3
        img_folder = "image_3"

    poses = dataset.poses
    poses_to_write = []
    dataset_len = len(dataset)

    for i in range(dataset_len):
        pcd = dataset.get_velo(i)
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(pcd[:, :3])

        legacy_pcd_obj = o3d.t.geometry.PointCloud.from_legacy(pcd_obj)
        transformed_pcd = legacy_pcd_obj.transform(cam_to_velo)

        if args.camera == "2":
            image = dataset.get_cam2(i)
        else:
            image = dataset.get_cam3(i)

        width, height = image.size
        depth_reproj = transformed_pcd.project_to_depth_image(
            width, height, cam_intrinsics, depth_max=10000.0
        )

        pcd_from_depth = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_reproj, cam_intrinsics, depth_max=10000.0
        )
        converted_pcd = o3d.t.geometry.PointCloud.to_legacy(pcd_from_depth)

        pcd_filename = "{:06d}.pcd".format(i)
        filepath = str(output_dir / pcd_filename)
        o3d.io.write_point_cloud(filepath, converted_pcd)

        cam0_to_velo = dataset.calib.T_cam0_velo

        pose_new = poses[i] @ cam0_to_velo @ np.linalg.inv(cam_to_velo)

        tx, ty, tz = pose_new[:3, 3]
        R = pose_new[:3, :3]
        qx, qy, qz, qw = mrob.geometry.so3_to_quat(R)

        str_pose = f"{tx} {ty} {tz} {qx} {qy} {qz} {qw}"
        poses_to_write.append(str_pose)

    with open(str(output_dir / "poses.txt"), "w") as file:
        for i in poses_to_write:
            file.write(i + "\n")

    img_src_folder = source_dir / "sequences" / args.sequence / img_folder
    img_out_folder = output_dir / img_folder

    copy_img_folder(img_src_folder, img_out_folder, dataset_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        This script converts data in KITTI format into the lib format using pykitti
        """
    )
    parser.add_argument("source_dir", help="Path to directory with KITTI dataset")
    parser.add_argument(
        "sequence",
        help="Folder with a specific sequence from the KITTI dataset. Select '00' if you want to select sequence number 00",
    )
    parser.add_argument(
        "camera", help="Choose '2' for the left camera or '3' for the right camera"
    )
    parser.add_argument(
        "output_dir", help="Path to the directory with converted data you want to save"
    )
    args = parser.parse_args()

    kitti_to_lib_format(args)
