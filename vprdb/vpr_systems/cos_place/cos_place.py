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
import cv2
import numpy as np
import torch
import torchvision

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

from vprdb.core import (
    Database,
    find_bounds_for_multiple_databases,
    match_two_databases,
    VoxelGrid,
)
from vprdb.vpr_systems.cos_place.model import GeoLocalizationNet
from vprdb.vpr_systems.cos_place.training import (
    CosPlaceTrainer,
    create_groups,
    DataModule,
)
from vprdb.vpr_systems.utils import make_deterministic


class CosPlace:
    def __init__(self, backbone: str, fc_output_dim: int, path_to_weights: str):
        self.backbone = backbone
        self.fc_output_dim = fc_output_dim
        self.path_to_weights = path_to_weights

        self.model = GeoLocalizationNet(backbone, fc_output_dim)
        model_state_dict = torch.load(self.path_to_weights)
        self.model.load_state_dict(model_state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_database_descriptors(self, database: Database):
        """
        Gets database RGB images CosPlace descriptors
        :param database: Database for getting descriptors
        :return: Descriptors for database images
        """
        self.model.eval()
        image_providers = database.color_images
        with torch.no_grad():
            all_descriptors = np.empty(
                (len(image_providers), self.fc_output_dim), dtype="float32"
            )
            for i, image in tqdm(
                enumerate(image_providers), total=len(image_providers)
            ):
                image_bgr = image.color_image
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                base_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                normalized_img = base_transform(image_rgb)
                normalized_img = normalized_img[None, :]
                descriptor = self.model(normalized_img.to(self.device))
                descriptor = descriptor.cpu().numpy()
                all_descriptors[i] = descriptor
        return all_descriptors

    def fine_tune_model(
        self,
        target_db: Database,
        valid_db: Database,
        train_db: Database,
        save_dir: str,
        voxel_size=0.3,
        random_resize=(480, 640),
        brightness=0.7,
        contrast=0.7,
        saturation=0.7,
        hue=0.5,
        random_resized_crop=0.5,
        num_workers=0,
        batch_size=8,
        lr=0.00001,
        classifiers_lr=0.01,
        patience=10,
        max_epochs=-1,
        seed=0,
    ) -> str:
        """
        Fine-tunes the CosPlace model for given target database
        :param target_db: The database for which the model will be fine-tuned
        :param valid_db: Validation database
        :param train_db: Training database
        :param save_dir: Directory for saving output model and log
        :param voxel_size: Voxel size for down sampling point clouds
        :return: Path to output model
        """
        min_bounds, max_bounds = find_bounds_for_multiple_databases(
            [target_db, valid_db, train_db]
        )
        voxel_grid = VoxelGrid(min_bounds, max_bounds, voxel_size)

        groups = create_groups(train_db, target_db, voxel_grid)
        groups_lens = [len(group_db) for group_db, _ in groups]
        val_matches = match_two_databases(valid_db, target_db, voxel_grid)

        make_deterministic(seed=seed)

        data = DataModule(
            groups,
            target_db,
            valid_db,
            val_matches,
            random_resize,
            brightness,
            contrast,
            saturation,
            hue,
            random_resized_crop,
            num_workers,
            batch_size,
        )
        train_model = CosPlaceTrainer(
            self.model,
            self.fc_output_dim,
            groups_lens,
            len(target_db),
            len(valid_db),
            lr,
            classifiers_lr,
            self.device,
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir, filename="best_model", save_weights_only=True
        )
        # start training
        trainer = Trainer(
            accelerator="auto",
            devices=[0],
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(monitor="R_1", mode="max", patience=patience),
                checkpoint_callback,
            ],
            default_root_dir=save_dir,
        )

        trainer.fit(
            train_model,
            datamodule=data,
        )

        # Transform weights to PyTorch format
        model_state_dict = torch.load(checkpoint_callback.best_model_path)
        model_state_dict = model_state_dict["state_dict"]
        new_model_state_dict = dict()
        for k in model_state_dict.keys():
            new_model_state_dict[k[6:]] = model_state_dict[k]
        torch.save(new_model_state_dict, checkpoint_callback.best_model_path)
        return checkpoint_callback.best_model_path
