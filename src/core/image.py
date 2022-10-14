from nptyping import NDArray, Shape, UInt8
from typing import Callable


class Image:
    def __init__(
        self,
        path_to_image: str,
        getter: Callable[[str], NDArray[Shape["*, *, 3"], UInt8]],
    ):
        self._path_to_image = path_to_image
        self._getter = getter

    def read(self) -> NDArray[Shape["*, *, 3"], UInt8]:
        return self._getter(self._path_to_image)
