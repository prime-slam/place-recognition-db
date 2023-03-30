#  Copyright (c) 2023, Musavian Mirfarid, Fakhriddin Tojiboev,
#  Ivan Moskalenko, Anastasiia Kornilova
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
import torchvision.transforms as T

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from vprdb.core import Database
from vprdb.vpr_systems.cos_place.training.dataset import TDataset


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_groups_dbs: list[tuple[Database, list[int]]],
        target_db: Database,
        valid_db: Database,
        valid_targets: list[int],
        random_resize,
        brightness,
        contrast,
        saturation,
        hue,
        random_resized_crop,
        num_workers,
        batch_size,
    ):
        super().__init__()
        self.resize = random_resize
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_resized_crop = random_resized_crop
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_transform = T.Compose(
            [
                T.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                ),
                T.RandomResizedCrop(
                    [self.resize[0], self.resize[1]],
                    scale=[1 - self.random_resized_crop, 1],
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.groups = {
            f"group_class_{ind}": TDataset(
                database=train_group_db, targets=targets, transform=self.train_transform
            )
            for ind, (train_group_db, targets) in enumerate(train_groups_dbs)
        }
        self.target_dataset = TDataset(database=target_db)
        self.valid_dataset = TDataset(database=valid_db, targets=valid_targets)

    def train_dataloader(self):
        train_dataloaders = {
            key: DataLoader(
                value,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            for key, value in self.groups.items()
        }
        return train_dataloaders

    def val_dataloader(self):
        db = DataLoader(
            self.target_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        query = DataLoader(
            self.valid_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        return [db, query]
