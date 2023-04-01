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
import itertools
import math
import numpy as np
import random
import torch
import torch.utils.data as data

from PIL import Image
from torch.utils.data import Dataset


class TDataset(Dataset):
    def __init__(
        self,
        db_paths,
        query_paths,
        associations,
        n_neg=5,
        transform=None,
        cached_queries=1000,
        bs=24,
        threads=8,
    ):
        # hyper-parameters
        self.n_neg = n_neg
        self.cached_queries = cached_queries

        # other
        self.transform = transform

        self.q_images = np.asarray(query_paths)
        self.db_images = np.asarray(db_paths)
        self.q_idx = np.arange(len(self.q_images))
        self.p_idx = np.expand_dims(associations, axis=1)

        # decide device type (important for triplet mining)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threads = threads
        self.bs = bs

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """

        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None, None, None, None, None

        query, positive, negatives, indices = zip(*batch)

        query = data.dataloader.default_collate(query)
        positive = data.dataloader.default_collate(positive)
        neg_counts = data.dataloader.default_collate([x.shape[0] for x in negatives])
        negatives = torch.cat(negatives, 0)
        indices = list(itertools.chain(*indices))

        return query, positive, negatives, neg_counts, indices

    def __len__(self):
        return len(self.triplets)

    def new_epoch(self):
        # find how many subset we need to do 1 epoch
        self.n_cache_subset = math.ceil(len(self.q_idx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.q_idx))

        # apply positive sampling of indices
        arr = random.choices(arr, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.n_cache_subset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache(self, net=None, outputdim=None):
        # reset triplets
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        # if net is None:
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        for q in qidxs:
            # get query idx
            qidx = self.q_idx[q]

            # get positives
            pidxs = self.p_idx[q]

            # choose a random positive (within positive range (default 10 m))
            pidx = np.random.choice(pidxs, size=1)[0]

            # get negatives
            while True:
                nidxs = np.random.choice(len(self.db_images), size=self.n_neg)

                # ensure that non of the choice negative images are within the negative range (default 25 m)
                if sum(np.in1d(nidxs, self.p_idx[q])) == 0:
                    break

            # package the triplet and target
            triplet = [qidx, pidx, *nidxs]
            target = [-1, 1] + [0] * len(nidxs)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

    def __getitem__(self, idx):
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        pidx = triplet[1]
        nidx = triplet[2:]

        # load images into triplet list
        query = self.transform(Image.open(self.q_images[qidx]))
        positive = self.transform(Image.open(self.db_images[pidx]))
        negatives = [self.transform(Image.open(self.db_images[idx])) for idx in nidx]
        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [qidx, pidx] + nidx
