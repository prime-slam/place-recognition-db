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
import torch.nn as nn

from torch.nn import Parameter


def cosine_sim(
    x1: torch.Tensor,
    x2: torch.Tensor,
    device: str = "cuda",
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    ip = torch.mm(x1, x2.t().to(device))
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2.to(device), 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        s: float = 30.0,
        m: float = 0.40,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.device = device
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight)
        one_hot = torch.zeros_like(cosine).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ", s="
            + str(self.s)
            + ", m="
            + str(self.m)
            + ")"
        )
