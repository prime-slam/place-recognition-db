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
import torch

from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from vprdb.vpr_systems.netvlad.training.t_dataset import TDataset
from vprdb.vpr_systems.netvlad.training.tools import humanbytes


def train_epoch(
    train_dataset,
    model,
    optimizer,
    criterion,
    encoder_dim,
    device,
    epoch_num,
    writer,
    bs,
    num_clusters,
    threads,
):
    train_dataset.new_epoch()

    epoch_loss = 0
    start_iter = 1  # keep track of batch iter across subsets for logging

    n_batches = (len(train_dataset.q_idx) + bs - 1) // bs

    for subIter in trange(
        train_dataset.n_cache_subset, desc="Cache refresh".rjust(15), position=1
    ):
        pool_size = encoder_dim
        pool_size *= num_clusters

        tqdm.write("====> Building Cache")
        train_dataset.update_subcache(model, pool_size)

        training_data_loader = DataLoader(
            dataset=train_dataset,
            num_workers=threads,
            batch_size=bs,
            shuffle=True,
            collate_fn=TDataset.collate_fn,
            pin_memory=device.type == "cuda",
        )

        tqdm.write("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
        tqdm.write("Cached:    " + humanbytes(torch.cuda.memory_cached()))

        model.train()
        for iteration, (query, positives, negatives, negCounts, indices) in enumerate(
            tqdm(
                training_data_loader,
                position=2,
                leave=False,
                desc="Train Iter".rjust(15),
            ),
            start_iter,
        ):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (n_query + n_pos + n_neg)
            if query is None:
                continue  # in case we get an empty batch

            B, C, H, W = query.shape
            n_neg = torch.sum(negCounts)
            data_input = torch.cat([query, positives, negatives])

            data_input = data_input.to(device)
            image_encoding = model.encoder(data_input)
            vlad_encoding = model.pool(image_encoding)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, n_neg])

            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(
                        vladQ[i : i + 1], vladP[i : i + 1], vladN[negIx : negIx + 1]
                    )

            loss /= n_neg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del data_input, image_encoding, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or n_batches <= 10:
                tqdm.write(
                    "==> Epoch[{}]({}/{}): Loss: {:.4f}".format(
                        epoch_num, iteration, n_batches, batch_loss
                    )
                )
                writer.add_scalar(
                    "Train/Loss", batch_loss, ((epoch_num - 1) * n_batches) + iteration
                )
                writer.add_scalar(
                    "Train/n_neg", n_neg, ((epoch_num - 1) * n_batches) + iteration
                )
                tqdm.write("Allocated: " + humanbytes(torch.cuda.memory_allocated()))
                tqdm.write("Cached:    " + humanbytes(torch.cuda.memory_cached()))

        start_iter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / n_batches

    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar("Train/AvgLoss", avg_loss, epoch_num)
