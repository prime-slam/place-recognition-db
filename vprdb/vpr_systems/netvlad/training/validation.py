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
import faiss
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vprdb.vpr_systems.netvlad.i_dataset import IDataset


def validate(
    eval_set,
    model,
    encoder_dim,
    device,
    writer,
    threads,
    cache_bs,
    num_clusters,
    epoch_num=0,
    write_tboard=False,
    pbar_position=0,
):
    eval_set_queries = IDataset(eval_set.q_images)
    eval_set_dbs = IDataset(eval_set.db_images)
    test_data_loader_queries = DataLoader(
        dataset=eval_set_queries,
        num_workers=threads,
        batch_size=cache_bs,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    test_data_loader_dbs = DataLoader(
        dataset=eval_set_dbs,
        num_workers=threads,
        batch_size=cache_bs,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    with torch.no_grad():
        tqdm.write("====> Extracting Features")
        pool_size = encoder_dim
        pool_size *= num_clusters
        q_feat = np.empty((len(eval_set_queries), pool_size), dtype=np.float32)
        db_feat = np.empty((len(eval_set_dbs), pool_size), dtype=np.float32)

        for feat, test_data_loader in zip(
            [q_feat, db_feat], [test_data_loader_queries, test_data_loader_dbs]
        ):
            for iteration, (input_data, indices) in enumerate(
                tqdm(
                    test_data_loader,
                    position=pbar_position,
                    leave=False,
                    desc="Test Iter".rjust(15),
                ),
                1,
            ):
                input_data = input_data.to(device)
                image_encoding = model.encoder(input_data)

                vlad_encoding = model.pool(image_encoding)
                feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                del input_data, image_encoding, vlad_encoding

    del test_data_loader_queries, test_data_loader_dbs

    tqdm.write("====> Building faiss index")
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(db_feat)

    tqdm.write("====> Calculating recall @ N")
    n_values = [1, 5, 10, 20, 50, 100]

    _, predictions = faiss_index.search(q_feat, max(n_values))

    gt = eval_set.p_idx
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(eval_set.q_idx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar("Val/Recall@" + str(n), recall_at_n[i], epoch_num)

    return all_recalls
