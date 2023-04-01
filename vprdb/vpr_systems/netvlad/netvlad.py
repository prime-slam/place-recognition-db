#  Copyright (c) 2023, Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, Tobias Fischer,
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
#
#  Significant part of our code is based on Patch-NetVLAD repository
#  (https://github.com/QVPR/Patch-NetVLAD)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from os import makedirs
from os.path import isfile, join
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange


from vprdb.core import (
    Database,
    find_bounds_for_multiple_databases,
    match_two_databases,
    VoxelGrid,
)
from vprdb.vpr_systems.netvlad.i_dataset import IDataset
from vprdb.vpr_systems.netvlad.model import get_backend, get_model, get_pca_encoding
from vprdb.vpr_systems.netvlad.training import (
    pca,
    save_checkpoint,
    TDataset,
    train_epoch,
    validate,
)
from vprdb.vpr_systems.utils import input_transform, make_deterministic


class NetVLAD:
    def __init__(self, path_to_weights: str):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.encoder_dim, self.encoder = get_backend()

        if isfile(path_to_weights):
            self.path_to_weights = path_to_weights
        else:
            raise FileNotFoundError(
                "=> no checkpoint found at '{}'".format(path_to_weights)
            )

        self.checkpoint = torch.load(
            self.path_to_weights, map_location=lambda storage, loc: storage
        )
        self.num_clusters = self.checkpoint["state_dict"]["pool.centroids"].shape[0]

    def get_database_descriptors(
        self,
        database: Database,
        resize=(480, 640),
        threads=0,
        batch_size=20,
        use_vladv2=False,
    ):
        """
        Gets database RGB images CosPlace descriptors
        :param database: Database for getting descriptors
        :param resize: Resizing before feature extraction
        :param threads: Number of workers
        :param batch_size: Size of a batch
        :param use_vladv2: If true, use vladv2 otherwise use vladv1
        :return: Descriptors for database images
        """
        num_pcs = self.checkpoint["state_dict"]["WPCA.0.bias"].shape[0]

        model = get_model(
            self.encoder,
            self.encoder_dim,
            self.num_clusters,
            use_vladv2,
            append_pca_layer=True,
            num_pcs=num_pcs,
        )
        model.load_state_dict(self.checkpoint["state_dict"])
        model = model.to(self.device)

        color_images_paths = [img.path for img in database.color_images]
        dataset = IDataset(color_images_paths, resize)
        test_data_loader = DataLoader(
            dataset=dataset,
            num_workers=threads,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.cuda,
        )
        model.eval()
        with torch.no_grad():
            db_feat = np.empty((len(dataset), num_pcs), dtype=np.float32)
            for iteration, (input_data, indices) in enumerate(
                tqdm(
                    test_data_loader,
                    position=1,
                    leave=False,
                    desc="Test Iter".rjust(15),
                ),
                1,
            ):
                indices_np = indices.detach().numpy()
                input_data = input_data.to(self.device)
                image_encoding = model.encoder(input_data)
                vlad_global = model.pool(image_encoding)
                vlad_global_pca = get_pca_encoding(model, vlad_global)
                db_feat[indices_np, :] = vlad_global_pca.detach().cpu().numpy()

        return db_feat

    def fine_tune_model(
        self,
        target_db: Database,
        valid_db: Database,
        train_db: Database,
        save_dir: str,
        voxel_size: float = 0.3,
        seed=42,
        add_pca=True,
        use_vladv2=False,
        optim_name="SGD",
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.001,
        lr_step=5,
        lr_gamma=0.5,
        margin=0.1,
        nNeg=5,
        cache_bs=20,
        bs=4,
        threads=0,
        max_epochs=100,
        eval_every=1,
        patience=5,
        n_features=10000,
        num_pcs=4096,
    ) -> str:
        """
        Fine-tunes the NetVLAD model for given target database
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
        train_targets = match_two_databases(train_db, target_db, voxel_grid)
        val_targets = match_two_databases(valid_db, target_db, voxel_grid)

        train_paths = [img.path for img in train_db.color_images]
        val_paths = [img.path for img in valid_db.color_images]
        db_paths = [img.path for img in target_db.color_images]

        make_deterministic(seed=seed)

        scheduler = None

        checkpoint = torch.load(
            self.path_to_weights, map_location=lambda storage, loc: storage
        )
        # Deleting WPCA layer
        del checkpoint["state_dict"]["WPCA.0.weight"]
        del checkpoint["state_dict"]["WPCA.0.bias"]

        model = get_model(self.encoder, self.encoder_dim, self.num_clusters, use_vladv2)
        model.load_state_dict(checkpoint["state_dict"])

        if optim_name == "ADAM":
            optimizer = optim.Adam(
                filter(lambda par: par.requires_grad, model.parameters()), lr=lr
            )
        elif optim_name == "SGD":
            optimizer = optim.SGD(
                filter(lambda par: par.requires_grad, model.parameters()),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step, gamma=lr_gamma
            )
        else:
            raise ValueError("Unknown optimizer: " + optim_name)

        criterion = nn.TripletMarginLoss(
            margin=(margin**0.5), p=2, reduction="sum"
        ).to(self.device)

        model = model.to(self.device)

        print("===> Loading dataset(s)")
        train_dataset = TDataset(
            db_paths,
            train_paths,
            train_targets,
            n_neg=nNeg,
            transform=input_transform(),
            bs=cache_bs,
            threads=threads,
        )

        validation_dataset = TDataset(
            db_paths,
            val_paths,
            val_targets,
            n_neg=nNeg,
            transform=input_transform(),
            bs=cache_bs,
            threads=threads,
        )

        print("===> Training query set:", len(train_dataset.q_idx))
        print("===> Evaluating on val set, query count:", len(validation_dataset.q_idx))
        print("===> Training model")

        writer = SummaryWriter(log_dir=save_dir)

        logdir = writer.file_writer.get_logdir()
        save_file_path = join(logdir, "checkpoints")
        makedirs(save_file_path)

        not_improved = 0
        best_score = 0
        for epoch in trange(
            1, max_epochs + 1, desc="Epoch number".rjust(15), position=0
        ):
            train_epoch(
                train_dataset,
                model,
                optimizer,
                criterion,
                self.encoder_dim,
                self.device,
                epoch,
                writer,
                bs,
                self.num_clusters,
                threads,
            )
            if scheduler is not None:
                scheduler.step(epoch)
            if (epoch % eval_every) == 0:
                recalls = validate(
                    validation_dataset,
                    model,
                    self.encoder_dim,
                    self.device,
                    writer,
                    threads,
                    cache_bs,
                    self.num_clusters,
                    epoch,
                    write_tboard=True,
                    pbar_position=1,
                )
                is_best = recalls[1] > best_score
                if is_best:
                    not_improved = 0
                    best_score = recalls[1]
                else:
                    not_improved += 1

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "recalls": recalls,
                        "best_score": best_score,
                        "not_improved": not_improved,
                        "optimizer": optimizer.state_dict(),
                        "parallel": False,
                    },
                    is_best,
                    save_file_path,
                )

                if patience > 0 and not_improved > (patience / int(eval_every)):
                    print(
                        "Performance did not improve for", patience, "epochs. Stopping."
                    )
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
        save_path = join(save_file_path, "model_best.pth.tar")
        print("Done")

        if add_pca:
            print("Adding PCA layer")
            model = get_model(
                self.encoder,
                self.encoder_dim,
                self.num_clusters,
                append_pca_layer=False,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(self.device)

            pool_size = self.encoder_dim
            pool_size *= self.num_clusters

            print("===> Loading PCA dataset(s)")

            if n_features > len(target_db):
                n_features = len(target_db)

            sampler = SubsetRandomSampler(
                np.random.choice(len(target_db), n_features, replace=False)
            )

            data_loader = DataLoader(
                dataset=IDataset(db_paths),
                num_workers=threads,
                batch_size=cache_bs,
                shuffle=False,
                pin_memory=self.cuda,
                sampler=sampler,
            )

            print("===> Do inference to extract features and save them.")

            model.eval()
            with torch.no_grad():
                tqdm.write("====> Extracting Features")

                db_feat = np.empty((len(data_loader.sampler), pool_size))
                print("Compute", len(db_feat), "features")

                for iteration, (input_data, indices) in enumerate(tqdm(data_loader)):
                    input_data = input_data.to(self.device)
                    image_encoding = model.encoder(input_data)
                    vlad_encoding = model.pool(image_encoding)
                    out_vectors = vlad_encoding.detach().cpu().numpy()
                    # this allows for randomly shuffled inputs
                    for idx, out_vector in enumerate(out_vectors):
                        db_feat[
                            iteration * data_loader.batch_size + idx, :
                        ] = out_vector

                    del input_data, image_encoding, vlad_encoding

            print("===> Compute PCA, takes a while")
            model_pca = pca(model, num_pcs, db_feat, pool_size)

            save_path = save_path.replace(".pth.tar", "_WPCA.pth.tar")

            torch.save({"state_dict": model_pca.state_dict()}, save_path)

            print("Done")

        return save_path
