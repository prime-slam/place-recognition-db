import networkx as nx

from vprdb.core import Database, VoxelGrid
from vprdb.reduction_methods.reduction_method import ReductionMethod


class DominatingSet(ReductionMethod):
    """
    Method constructs a graph where the images are vertices that
    are connected if the IoU is greater than the
    given threshold. Then it finds a minimal subset of graph vertices so
    that every vertex is either in the set or is connected to a
    vertex from it.
    """

    def __init__(self, threshold: float = 0.3, voxel_size: float = 0.3):
        """
        Constructs DominatingSet reduction method
        :param threshold: Threshold value for IoU
        :param voxel_size: The value indicating which IoU value will be enough
        to consider the point clouds as overlapping
        """
        self.threshold = threshold
        self.voxel_size = voxel_size

    def reduce(self, db: Database) -> Database:
        min_bounds, max_bounds = db.bounds
        voxel_grid = VoxelGrid(min_bounds, max_bounds, self.voxel_size)
        # Determine which frames cover a particular voxel
        voxel_to_frames_dict = dict()
        # Frame size is the number of voxels it covers
        frames_sizes = []
        for i, pose in enumerate(db.trajectory):
            pcd = db.point_clouds[i].point_cloud.transform(pose)
            down_sampled_pcd = voxel_grid.voxel_down_sample(pcd)
            points = down_sampled_pcd.points
            frames_sizes.append(len(points))
            for point in points:
                voxel_index = voxel_grid.get_voxel_index(point)
                if voxel_index in voxel_to_frames_dict:
                    voxel_to_frames_dict[voxel_index].append(i)
                else:
                    voxel_to_frames_dict[voxel_index] = [i]

        intersections = dict()
        for covering_frames in voxel_to_frames_dict.values():
            for i, frame_1 in enumerate(covering_frames):
                for frame_2 in covering_frames[i + 1 :]:
                    intersection = tuple(sorted((frame_1, frame_2)))
                    if intersection in intersections:
                        intersections[intersection] += 1
                    else:
                        intersections[intersection] = 1

        IoUs = dict.fromkeys(intersections.keys())
        for (frame_1, frame_2), intersection in intersections.items():
            IoUs[(frame_1, frame_2)] = intersection / (
                frames_sizes[frame_1] + frames_sizes[frame_2] - intersection
            )

        G = nx.Graph()
        G.add_nodes_from(range(len(db)))
        for (fr1, fr2), IoU in IoUs.items():
            if IoU > self.threshold:
                G.add_edge(fr1, fr2)
        result_indices = list(nx.dominating_set(G))
        result_indices.sort()
        new_rgb = [db.color_images[i] for i in result_indices]
        new_point_clouds = [db.point_clouds[i] for i in result_indices]
        new_traj = [db.trajectory[i] for i in result_indices]
        return Database(new_rgb, new_point_clouds, new_traj)

    reduce.__doc__ = ReductionMethod.reduce.__doc__
