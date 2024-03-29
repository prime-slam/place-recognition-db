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
import torchvision

from vprdb.vpr_systems.cos_place.model.layers import Flatten, L2Norm, GeM

CHANNELS_NUM_IN_LAST_CONV = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "vgg16": 512,
}


class GeoLocalizationNet(torch.nn.Module):
    def __init__(self, backbone, fc_output_dim):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = torch.nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            torch.nn.Linear(features_dim, fc_output_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)

        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[
            :-2
        ]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False

    backbone = torch.nn.Sequential(*layers)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim
