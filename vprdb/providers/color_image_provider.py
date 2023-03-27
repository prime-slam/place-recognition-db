import cv2

from dataclasses import dataclass
from nptyping import NDArray, Shape, UInt8
from pathlib import Path


@dataclass(frozen=True)
class ColorImageProvider:
    path: Path

    @property
    def color_image(self) -> NDArray[Shape["*, *, 3"], UInt8]:
        return cv2.imread(str(self.path), cv2.IMREAD_COLOR)
