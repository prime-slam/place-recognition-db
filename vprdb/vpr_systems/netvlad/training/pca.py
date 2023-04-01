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

from scipy.sparse.linalg import eigs

from vprdb.vpr_systems.netvlad.model import Flatten, L2Norm


def pca(model, num_pcs, db_feat, pool_size):
    # translated from MATLAB:
    # - https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m
    # - https://github.com/Relja/netvlad/blob/master/addPCA.m

    # assumes db_feat = nvectors x ndims
    db_feat = db_feat.T  # matlab code is ndims x nvectors, so transpose

    n_points = db_feat.shape[1]
    n_dims = db_feat.shape[0]

    if num_pcs is None:
        num_pcs = n_dims

    # Subtract mean
    mu = np.mean(db_feat, axis=1)
    db_feat = (db_feat.T - mu).T

    assert num_pcs < n_dims

    if n_dims <= n_points:
        do_dual = False
        x2 = np.matmul(db_feat, db_feat.T) / (n_points - 1)
    else:
        do_dual = True
        x2 = np.matmul(db_feat.T, db_feat) / (n_points - 1)

    if num_pcs < x2.shape[0]:
        print("Compute {} eigenvectors".format(num_pcs))
        lams, u = eigs(x2, num_pcs)
    else:
        print("Compute eigenvectors")
        lams, u = np.linalg.eig(x2)

    assert np.all(np.isreal(lams)) and np.all(np.isreal(u))
    lams = np.real(lams)
    u = np.real(u)

    sort_indices = np.argsort(lams)[::-1]
    lams = lams[sort_indices]
    u = u[:, sort_indices]

    if do_dual:
        diag = np.diag(1.0 / np.sqrt(np.maximum(lams, 1e-9)))
        utimesdiag = np.matmul(u, diag)
        u = np.matmul(db_feat, utimesdiag / np.sqrt(n_points - 1))

    u = u[:, :num_pcs]
    lams = lams[:num_pcs]

    print("===> Add PCA Whiten")
    u = np.matmul(u, np.diag(np.divide(1.0, np.sqrt(lams + 1e-9))))
    pca_str = "WPCA"

    utmu = np.matmul(u.T, mu)

    pca_conv = nn.Conv2d(pool_size, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
    pca_conv.weight = nn.Parameter(
        torch.from_numpy(np.expand_dims(np.expand_dims(u.T, -1), -1))
    )
    pca_conv.bias = nn.Parameter(torch.from_numpy(-utmu))

    model.add_module(pca_str, nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))
    return model
