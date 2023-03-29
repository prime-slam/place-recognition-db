#  Copyright (c) 2023, Gabriele Berton, Carlo Masone, Barbara Caputo,
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
#  Significant part of our code is based on CosPlace repository
#  (https://github.com/gmberton/CosPlace)
import torch


def gem(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    return torch.nn.functional.avg_pool2d(
        x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
    ).pow(1.0 / p)


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2.0, dim=self.dim)
