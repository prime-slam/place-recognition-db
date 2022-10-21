import numpy as np

from src.core import Database
from src.reduction_methods.reduction_method import ReductionMethod


class DistanceVectorSeq(ReductionMethod):
    """
    The method calculates the distance between the previously taken pose and the current pose.
    If the distance is greater than the threshold value,
    the pose is returned and the status of the last pose is updated
    """

    def __init__(self, distance_threshold: float):
        """
        Constructs DistanceVector reduction method
        :param distance_threshold: Threshold value for distance
        """
        self.distance_threshold = distance_threshold

    def reduce(self, db: Database) -> Database:
        traj = db.trajectory
        new_traj = [traj[0]]
        new_rgb = [db.images[0]]
        new_pcds = [db.pcds[0]]
        dist_point = traj[0][:3, 3]
        for i, trans_matrix in enumerate(traj):
            point = trans_matrix[:3, 3]
            distance = np.linalg.norm(point - dist_point)
            if distance > self.distance_threshold:
                new_traj.append(trans_matrix)
                new_rgb.append(db.images[i])
                new_pcds.append(db.pcds[i])
                dist_point = point

        return Database(new_traj, new_rgb, new_pcds)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
