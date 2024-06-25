import open3d as o3d
from scipy.spatial import KDTree
import numpy as np
from collections import deque
from utils import compute_3d_iou
from utils import time_it
from utils import averageQuaternions, weightedAverageQuaternions
import rospy
from scipy.spatial.transform import Rotation as R
import copy
import threading


TIME_TO_BE_CONFIRMED = 0.2
EXPIRY_TIME_MOVING_OBJECTS = 2.0
STD_THR = 0.5
VOXEL_SIZE = 0.05
MOVING_CLASSES = ["person"]


def exponential_weights(length, decay_rate=0.1):
    weights = np.exp(-decay_rate * np.arange(length))
    return weights / np.sum(weights)


def linear_weights(length):
    weights = np.linspace(1, 0, length)
    return weights / np.sum(weights)


class Obj:
    """Object class with historical data and spatial indexing using KDTree."""

    __slots__ = [
        "pcd",
        "label",
        "score",
        "bbox",
        "id",
        "centroid",
        "last_seen",
        "first_seen",
        "is_confirmed",
        "centroids",
        "quaternions",
    ]

    def __init__(self, id, points, label, score, stamp):
        self.id = id
        self.label = label
        self.score = score
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.last_seen = stamp
        self.first_seen = stamp
        self.is_confirmed = False
        self.centroids = np.asarray(self.pcd.points).mean(axis=0).reshape(1, 3)

        r = self.pcd.get_rotation_matrix_from_xyz((0, 0, 0))
        self.quaternions = np.array([R.from_matrix(r).as_quat()])
        self.compute()

    def update(self, other: "Obj"):
        """Updates the object's points and score and stores the historical state."""

        # saving other's centroid before registration
        other_centroid = np.asarray(other.pcd.points).mean(axis=0)
        self.centroids = np.concatenate((self.centroids, [other_centroid]), axis=0)
        #
        ## saving other's quaternion before registration
        r = other.pcd.get_rotation_matrix_from_xyz((0, 0, 0))
        self.quaternions = np.concatenate(
            (self.quaternions, [R.from_matrix(r).as_quat()]), axis=0
        )
        #
        try:
            other = self.register_pointcloud_to_self(other)
        except Exception as e:
            rospy.logwarn(f"Error in registering point clouds: {e}")
            return
        #
        self.last_seen = max(self.last_seen, other.last_seen)

        # Euristics: an object is confirmed if it has been seen multiple times
        if not self.is_confirmed:
            self.is_confirmed = self.last_seen - self.first_seen >= rospy.Duration(
                TIME_TO_BE_CONFIRMED
            )

        # Use more sophisticated logic for updating point cloud data
        if self.label == other.label and self.label in MOVING_CLASSES:
            self.pcd.points = other.pcd.points
        else:
            old_points = np.asarray(self.pcd.points)
            new_points = np.asarray(other.pcd.points)
            combined_points = np.concatenate((old_points, new_points), axis=0)
            self.pcd.points = o3d.utility.Vector3dVector(combined_points)

            weights = linear_weights(len(self.quaternions))
            centroid = np.average(self.centroids, axis=0, weights=weights)
            quaternion = weightedAverageQuaternions(self.quaternions, weights)
            rot_matrix = R.from_quat(quaternion).as_matrix()
            self.pcd.translate(centroid, relative=False)
            self.pcd.rotate(rot_matrix)

        self.compute()
        self.label = other.label if other.score > self.score else self.label
        self.score = max(self.score, other.score)

    def register_pointcloud_to_self(self, other: "Obj", threshold=0.2):
        """Aligns other point cloud to self using ICP registration."""
        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                other.pcd,
                self.pcd,
                threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            if reg_p2p.inlier_rmse > threshold:
                raise ValueError("High registration error, skipping this frame")

            transformation = reg_p2p.transformation

            # Transform the source point cloud
            other.pcd.transform(transformation)
            return other
        except Exception as e:
            print(f"ICP registration failed: {e}")
            raise e

    def compute_z_oriented_bounding_box(self, pcd):
        # Compute the mean of the points
        if not pcd.points:
            return None
        mean = np.mean(np.asarray(pcd.points), axis=0)

        # Compute PCA on the XY components only
        xy_points = np.asarray(pcd.points)[:, :2] - mean[:2]
        cov_matrix = np.dot(xy_points.T, xy_points) / len(xy_points)
        eigvals, eigvecs = np.linalg.eig(cov_matrix)

        # Align primary component with the X-axis
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        R = pcd.get_rotation_matrix_from_xyz((0, 0, -angle))
        pcd.rotate(R, center=mean)

        # Compute the axis-aligned bounding box of the rotated point cloud
        aabb = pcd.get_axis_aligned_bounding_box()
        # convert aabb to oriented bbox
        aabb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            aabb
        )

        # Apply inverse rotation to the bounding box
        aabb.rotate(R.T, center=mean)

        return np.asarray(aabb.get_box_points())

    def compute(self):
        self.pcd = self.pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        self.pcd, _ = self.pcd.remove_radius_outlier(nb_points=16, radius=0.5)
        # clean, _ = self.pcd.remove_radius_outlier(nb_points=10, radius=0.5)

        clean, _ = self.pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
        # clean = self.pcd
        if clean.points is None or len(clean.points) <= 10:
            self.bbox = np.zeros((8, 3))
            self.centroid = np.zeros(3)
            return

        self.bbox = self.compute_z_oriented_bounding_box(clean)
        self.centroid = np.asarray(clean.points).mean(axis=0)


class World:
    """World class that uses a KDTree for efficient spatial management of objects."""

    def __init__(self):
        self.objects = {}
        self.kdtree = None
        self.points_list = []
        self.id2index = {}
        self.index2id = {}
        self.lock = threading.Lock()  # thread safe for deleting operations
        if o3d.core.cuda.is_available():
            print("CUDA is available in Open3D.")
        else:
            print("CUDA is not available in Open3D.")

    # @time_it
    def add_object(self, obj):
        """Adds a new object to the world and updates the KDTree."""
        with self.lock:
            self.objects[obj.id] = obj
            self.points_list.append(obj.centroid)  # Using centroid of the bounding box
            self.id2index[obj.id] = len(self.points_list) - 1
            self.index2id[len(self.points_list) - 1] = obj.id
            self._rebuild_kdtree()

    # @time_it
    def update_object(self, obj):
        """Updates an existing object in the world."""
        self.objects[obj.id].update(obj)
        self.points_list[self.id2index[obj.id]] = self.objects[obj.id].centroid
        self._rebuild_kdtree()

    # @time_it
    def remove_objects(self, obj_ids):
        """Removes an object from the world."""
        with self.lock:
            for obj_id in obj_ids:
                if obj_id in self.objects:
                    self.objects.pop(obj_id)

            self.points_list = []
            self.id2index = {}
            self.index2id = {}

            for i, (id, obj) in enumerate(self.objects.items()):
                self.id2index[id] = i
                self.index2id[i] = id
                self.points_list.append(obj.centroid)

            self._rebuild_kdtree()

    def override_object(self, old_id, obj):
        old_obj = self.objects[old_id]
        rospy.loginfo(
            f"Overriding object {old_id}:{old_obj.label} with {obj.id}:{obj.label}"
        )
        self.objects[obj.id].update(old_obj)
        self.remove_objects([old_id])

    def get_objects(self):
        """Returns all objects in the world."""
        return list([obj for obj in self.objects.values() if obj.is_confirmed])

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
            inf_index = np.where(self.points_list == np.repeat(np.inf, 3))
            none_index = np.where(self.points_list == np.repeat(np.nan, 3))
            #print(inf_index, none_index)
            try:
                self.kdtree = KDTree(np.array(self.points_list))
            except:
                print(self.points_list, inf_index, none_index)

    # @time_it
    def get_world_id(self, obj: Obj, distance_thr=1, iou_thr=0.05):
        """
        Checks if the object already exists in the world by comparing 3D IoU and label of close objects
        args:
            obj: Object to be checked
            distance_thr: Distance threshold for querying close objects (in meters)
            iou_thr: IoU threshold for considering two objects as the same
        Returns: The ID of the object in the world if it exists, otherwise the object's ID.
        """
        close_objects = self.query_by_distance(obj.centroid, distance_thr)
        for close_obj in close_objects:
            # distance = np.median(obj.pcd.compute_point_cloud_distance(close_obj.pcd))
            if (
                obj.id != close_obj.id
                and obj.label == close_obj.label
                and abs(obj.last_seen - close_obj.last_seen).to_sec() > 5.0
            ):
                # print(f"Distance: {distance} between {obj.id}:{obj.label} and {close_obj.id}:{close_obj.label}")
                if compute_3d_iou(obj.bbox, close_obj.bbox) > iou_thr:
                    return close_obj.id

        return obj.id

    # @time_it
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

    def clean_up(self):
        """
        Removes moving object that are not currently tracked or objects which Pointcloud has a std greater than a thr.
        """
        to_remove = {}
        for obj in self.objects.values():
            if (
                    obj.label in MOVING_CLASSES
                    and obj.last_seen.to_sec()
                    < rospy.Time.now().to_sec() - EXPIRY_TIME_MOVING_OBJECTS
            ):
                to_remove[obj.id] = "Expired"
            
            std = np.std(np.asarray(obj.pcd.points), axis=0).max()
            if  std > STD_THR:
                to_remove[obj.id] = f"STD above Thr: {std}> {STD_THR}"

        if len(to_remove) > 0:
            print("To remove: ", to_remove)
            self.remove_objects(list(to_remove.keys()))

    def get_kdtree_centroids(self):
        return self.points_list
