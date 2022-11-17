import cv2

from dataclasses import dataclass
from nptyping import NDArray, Shape, UInt8


@dataclass(frozen=True)
class ImageProvider:
    path_to_image: str

    @property
    def image(self) -> NDArray[Shape["*, *, 3"], UInt8]:
        return cv2.imread(self.path_to_image, cv2.IMREAD_COLOR)
