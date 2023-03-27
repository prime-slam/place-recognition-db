from vprdb.core import Database
from vprdb.reduction_methods.reduction_method import ReductionMethod


class EveryNth(ReductionMethod):
    """The method returns every N-th item from DB"""

    def __init__(self, n: int):
        """
        Constructs EveryNth reduction method
        :param n: Step of method
        """
        self.n = n

    def reduce(self, db: Database) -> Database:
        new_traj = db.trajectory[:: self.n]
        new_color = db.color_images[:: self.n]
        new_point_clouds = db.point_clouds[:: self.n]
        return Database(new_color, new_point_clouds, new_traj)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
