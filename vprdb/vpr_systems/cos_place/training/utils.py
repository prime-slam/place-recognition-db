#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from vprdb.core import calculate_iou, Database, match_two_databases, VoxelGrid


def create_groups(train_database: Database, target_db: Database, voxel_grid: VoxelGrid):
    classes_dict = dict.fromkeys(range(len(target_db)))
    for i in range(len(target_db)):
        classes_dict[i] = []

    train_database_matches = match_two_databases(train_database, target_db, voxel_grid)
    for i, match in enumerate(train_database_matches):
        classes_dict[match].append(i)

    classes = set(range(len(target_db)))
    groups = list()
    while len(classes) > 0:
        new_group = set()
        for class_ in classes:
            available_to_add = True
            for inner_class in new_group:
                pose, pcd = (
                    target_db.trajectory[class_],
                    target_db.point_clouds[class_],
                )
                pose_inner, pcd_inner = (
                    target_db.trajectory[inner_class],
                    target_db.point_clouds[inner_class],
                )
                pcd_1 = pcd.point_cloud.transform(pose)
                pcd_2 = pcd_inner.point_cloud.transform(pose_inner)
                iou = calculate_iou(pcd_1, pcd_2, voxel_grid)
                if iou > 0:
                    available_to_add = False
                    break
            if available_to_add:
                new_group.add(class_)
        classes = classes.difference(new_group)
        if len(new_group) > 1:
            groups.append(new_group)

    groups_with_images = []
    for group in groups:
        train_group_indices = []
        train_group_targets = []
        for class_ in group:
            train_group_indices.extend(classes_dict[class_])
            train_group_targets.extend([class_] * len(classes_dict[class_]))
        train_group_rgb = [train_database.color_images[i] for i in train_group_indices]
        train_group_pcds = [train_database.point_clouds[i] for i in train_group_indices]
        train_group_traj = [train_database.trajectory[i] for i in train_group_indices]
        train_group_db = Database(train_group_rgb, train_group_pcds, train_group_traj)
        if len(train_group_db) > len(target_db):
            groups_with_images.append((train_group_db, train_group_targets))
    return groups_with_images
