#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
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
import cv2

from dataclasses import dataclass
from nptyping import NDArray, Shape, UInt8
from pathlib import Path


@dataclass(frozen=True)
class ColorImageProvider:
    """Color image provider is a wrapper for color images"""

    path: Path
    """ Path to file on hard drive"""

    @property
    def color_image(self) -> NDArray[Shape["*, *, 3"], UInt8]:
        """Returns image in OpenCV format"""
        return cv2.imread(str(self.path), cv2.IMREAD_COLOR)
