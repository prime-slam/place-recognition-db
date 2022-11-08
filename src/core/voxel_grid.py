import numpy as np

from dataclasses import dataclass
from nptyping import Float, NDArray, Shape


@dataclass
class VoxelGrid:
    """Voxel grid with given boundaries and voxel size"""

    min_bounds: NDArray[Shape["3"], Float]
    max_bounds: NDArray[Shape["3"], Float]
    voxel_size: float

    def get_voxel_index(
        self, point: NDArray[Shape["3"], Float]
    ) -> tuple[int, int, int]:
        """
        The method gets the voxel index for a given point
        Implemented according to the corresponding Open3D method
        :param point: Point to get its corresponding voxel index
        :return: Voxel index
        """
        ref_coord = (point - self.min_bounds) / self.voxel_size
        x_index, y_index, z_index = np.floor(ref_coord).astype(int)
        return x_index, y_index, z_index
