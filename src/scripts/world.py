from scipy.spatial import KDTree
import numpy as np
from collections import deque
from utils import compute_3d_iou
import open3d as o3d


class Obj:
    """Object class with historical data and spatial indexing using KDTree."""

    __slots__ = ["pcd", "label", "score", "bbox", "id"]

    def __init__(self, id, points, label, score):
        self.id = id
        self.label = label
        self.score = score
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
        self.pcd, _ = self.pcd.remove_radius_outlier(nb_points=15, radius=0.05)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox = np.asarray(self.bbox.get_box_points())
        

    def update(self, other: "Obj"):
        """Updates the object's points and score and stores the historical state."""
        points = np.concatenate([self.pcd.points, other.pcd.points], axis = 0)
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.02)
        self.pcd, _ = self.pcd.remove_radius_outlier(nb_points=15, radius=0.05)

        self.bbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox = np.asarray(self.bbox.get_box_points())

        self.label = other.label if other.score > self.score else self.label
        self.score = np.max([self.score, other.score])


class World:
    """World class that uses a KDTree for efficient spatial management of objects."""

    def __init__(self):
        self.objects = {}
        self.kdtree = None
        self.points_list = []
        self.id2index = {}
        self.index2id = {}

    def add_object(self, obj):
        """Adds a new object to the world and updates the KDTree."""
        self.objects[obj.id] = obj
        self.points_list.append(
            obj.pcd.points.mean(axis=0)
        )  # Using centroid of the bounding box
        self.id2index[obj.id] = len(self.points_list) - 1
        self.index2id[len(self.points_list) - 1] = obj.id
        self._rebuild_kdtree()

    def update_object(self, obj):
        """Updates an existing object in the world."""
        self.objects[obj.id].update(obj)
        self.points_list[self.id2index[obj.id]] = self.objects[obj.id].pcd.points.mean(
            axis=0
        )
        self._rebuild_kdtree()

    def remove_object(self, obj_id):
        """Removes an object from the world."""
        if obj_id in self.objects:
            self.objects.pop(obj_id)
            self.points_list = []
            self.id2index = {}
            self.index2id = {}

            for i, (id, obj) in enumerate(self.objects.items()):
                self.id2index[id] = i
                self.index2id[i] = id
                self.points_list.append(obj.pcd.points.mean(axis=0))

            self._rebuild_kdtree()

    def override_object(self, old_id, obj):
        """ """
        old_obj = self.objects[old_id]
        print(f"Overriding object {old_id}:{old_obj.label} with {obj.id}:{obj.label}")
        self.objects[obj.id].update(old_obj)
        self.remove_object(old_id)

    def get_objects(self):
        """Returns all objects in the world."""
        return list(self.objects.values())

    def query_by_distance(self, point, threshold):
        """Queries objects within a certain distance threshold."""
        if self.kdtree is not None:
            indexes = self.kdtree.query_ball_point(point, r=threshold)
            return [
                self.objects[obj_id]
                for obj_id in np.array(list(self.objects.keys()))[indexes]
            ]
        return []

    def _rebuild_kdtree(self):
        if len(self.points_list) > 0:
            self.kdtree = KDTree(np.array(self.points_list))

    def get_world_id(self, obj: Obj, distance_thr=2, iou_thr=0.0):
        """
        Checks if the object already exists in the world by comparing 3D IoU and label of close objects
        args:
            obj: Object to be checked
            distance_thr: Distance threshold for querying close objects (in meters)
            iou_thr: IoU threshold for considering two objects as the same
        Returns: The ID of the object in the world if it exists, otherwise the object's ID.
        """
        close_objects = self.query_by_distance(obj.pcd.points.mean(axis=0), distance_thr)
        for close_obj in close_objects:
            if (
                obj.id != close_obj.id
                and obj.label == close_obj.label
                and compute_3d_iou(obj.bbox, close_obj.bbox) > iou_thr
            ):
                return close_obj.id

        return obj.id

    def manage_object(self, obj: Obj):

        if obj.id in self.objects:
            self.update_object(obj)
            # taking the updated object
            obj = self.objects[obj.id]
        else:
            self.add_object(obj)

        world_id = self.get_world_id(obj)
        if world_id != obj.id:
            self.override_object(world_id, obj)

        return self.objects[obj.id]
