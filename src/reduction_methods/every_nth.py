from src.reduction_methods.reduction_method import ReductionMethod
from src.database import Database


class EveryNth(ReductionMethod):
    """The method returns every N-th pose from trajectory"""

    def __init__(self, n: int):
        """
        Constructs EveryNth reduction method
        :param n: Step of method
        """
        self.n = n

    def reduce(self, db: Database) -> Database:
        new_traj = db.trajectory[:: self.n]
        new_rgb = db.rgb_images_paths[:: self.n]
        new_pcds = db.pcds_paths[:: self.n]
        return Database(new_traj, new_rgb, new_pcds, db.rgb_getter, db.pcd_getter)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
