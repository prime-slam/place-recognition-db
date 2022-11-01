from dataclasses import dataclass
from nptyping import Float, NDArray, Shape


@dataclass
class VoxelGrid:
    """Voxel grid with given boundaries and voxel size"""

    min_bounds: NDArray[Shape["3"], Float]
    max_bounds: NDArray[Shape["3"], Float]
    voxel_size: float
