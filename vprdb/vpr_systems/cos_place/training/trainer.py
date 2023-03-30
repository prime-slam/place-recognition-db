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
import faiss
import numpy as np
import torch

from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from vprdb.vpr_systems.cos_place.model import GeoLocalizationNet, MarginCosineProduct


class CosPlaceTrainer(LightningModule):
    def __init__(
        self,
        model_to_finetune: GeoLocalizationNet,
        fc_output_dim: int,
        groups_lens: list[int],
        target_db_len: int,
        valid_db_len: int,
        lr: float,
        classifiers_lr: float,
        cos_place_device: torch.device,
    ):
        super().__init__()
        self.groups_lens = groups_lens
        self.fc_output_dim = fc_output_dim
        self.cos_place_device = cos_place_device

        self.criterion = CrossEntropyLoss()
        self.classifiers = [
            MarginCosineProduct(self.fc_output_dim, group_len, self.cos_place_device)
            for group_len in self.groups_lens
        ]

        self.model = model_to_finetune
        self.model.train()

        self.lr = lr
        self.classifiers_lr = classifiers_lr

        self.db_descriptors = np.empty(
            (target_db_len, self.fc_output_dim), dtype="float32"
        )
        self.db_groundtruth_labels = np.empty((target_db_len, 1))

        self.query_dataset_descriptors = np.empty(
            (valid_db_len, self.fc_output_dim), dtype="float32"
        )
        self.query_dataset_groundtruth_labels = np.empty((valid_db_len, 1))
        self.automatic_optimization = False

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        optimizers_list = self.optimizers()
        model_opt, classifiers_opt = optimizers_list[0], optimizers_list[1:]

        current_group_num = self.current_epoch % len(self.groups_lens)

        images, _, targets = batch[f"group_class_{current_group_num}"]
        images, targets = images.to(self.device), targets.to(self.device)

        descriptors = self.model(images)
        output = self.classifiers[current_group_num](descriptors, targets)
        loss = self.criterion(output, targets)
        model_opt.zero_grad()
        classifiers_opt[current_group_num].zero_grad()
        self.manual_backward(loss)
        model_opt.step()
        classifiers_opt[current_group_num].step()

    def validation_step(self, batch, batch_idx, dataloader_idx):
        images, indices, targets = batch
        descriptors = self.model(images)
        descriptors = descriptors.cpu().numpy()
        if dataloader_idx == 0:
            self.db_descriptors[indices.cpu().numpy(), :] = descriptors
            self.db_groundtruth_labels[indices.cpu().numpy(), :] = np.expand_dims(
                targets.cpu().numpy(), axis=1
            )
        else:
            self.query_dataset_descriptors[indices.cpu().numpy(), :] = descriptors
            self.query_dataset_groundtruth_labels[
                indices.cpu().numpy(), :
            ] = np.expand_dims(targets.cpu().numpy(), axis=1)

    def on_validation_epoch_end(self):
        faiss_index = faiss.IndexFlatL2(self.fc_output_dim)
        faiss_index.add(self.db_descriptors)
        _, predictions = faiss_index.search(self.query_dataset_descriptors, 1)
        recall = self.compute_recall(predictions)
        self.log("R_1", recall, logger=True, prog_bar=True)

    def compute_recall(self, predictions):
        recall = 0
        for query_index, preds in enumerate(predictions):
            if (
                self.db_groundtruth_labels[preds[0]]
                == self.query_dataset_groundtruth_labels[query_index]
            ):
                recall += 1
        # Divide by queries_num and multiply by 100, so the recalls are in percentages
        recall = recall / len(self.query_dataset_descriptors) * 100
        return recall

    def configure_optimizers(self):
        model_optimizer = Adam(self.parameters(), lr=self.lr)
        classifiers_optimizers = [
            Adam(classifier.parameters(), lr=self.classifiers_lr)
            for classifier in self.classifiers
        ]

        return [model_optimizer] + classifiers_optimizers
