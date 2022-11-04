from dataclasses import dataclass
from nptyping import NDArray, Shape, UInt8
from typing import Callable


@dataclass(frozen=True)
class ImageProvider:
    path_to_image: str
    getter: Callable[[str], NDArray[Shape["*, *, 3"], UInt8]]

    @property
    def image(self) -> NDArray[Shape["*, *, 3"], UInt8]:
        return self.getter(self.path_to_image)
