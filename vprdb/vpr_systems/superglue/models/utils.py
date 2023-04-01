#  Copyright (c) 2023, Magic Leap, Ivan Moskalenko, Anastasiia Kornilova
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
#  Significant part of our code is based on SuperGluePretrainedNetwork repository
#  (https://github.com/magicleap/SuperGluePretrainedNetwork)
import cv2
import torch


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def read_image(path, device, resize, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if resize_float:
        image = cv2.resize(image.astype("float32"), resize)
    else:
        image = cv2.resize(image, resize).astype("float32")

    inp = frame2tensor(image, device)
    return inp
