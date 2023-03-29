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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vprdb.core import Database


class TDataset(Dataset):
    def __init__(self, database: Database, targets: list[int] = None, transform=None):
        super().__init__()

        if targets is None:
            targets = list(range(len(database)))

        if len(database) != len(targets):
            raise ValueError(
                "The length of the targets and the database must be the same"
            )

        self.database = database
        self.targets = targets

        if transform is not None:
            self.base_transform = transform
        else:
            self.base_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, index):
        image_path, cls = self.database.color_images[index].path, self.targets[index]
        pil_img = Image.open(image_path).convert("RGB")
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index, int(cls)

    def __len__(self):
        return len(self.database)
