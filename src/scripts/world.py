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
from s_map.msg import Object, ObjectList
from compas.geometry import oriented_bounding_box_numpy


TIME_TO_BE_CONFIRMED = 1.0
EXPIRY_TIME_MOVING_OBJECTS = 0.0
STD_THR = 0.6
VOXEL_SIZE = 0.03
MOVING_CLASSES = []
IOU_THR = 0.0


def exponential_weights(length, decay_rate=0.1):
    weights = np.exp(-decay_rate * np.arange(length))
    return weights / np.sum(weights)


def linear_weights(length):
    weights = np.linspace(1, 0, length)
    return weights / np.sum(weights)


device = (
    o3d.core.Device("CUDA:0")
    if o3d.core.cuda.is_available()
    else o3d.core.Device("CPU:0")
)


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
        "device",
    ]

    def __init__(self, id, points, label, score, stamp):
        self.id = id
        self.label = label
        self.score = score
        self.device = device
        self.pcd = o3d.t.geometry.PointCloud(device=self.device)

        points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points should be a 2D array with shape (N, 3)")

        # If points are too much, randomly sample n points
        if len(points) > 3000:
            idx = np.random.choice(len(points), 3000, replace=False)
            points = points[idx]

        try:
            self.pcd.point.positions = o3d.core.Tensor(
                points, o3d.core.float32, self.device
            )
        except Exception as e:
            rospy.logerr(f"Error creating tensor for point cloud positions: {e}")
            raise ValueError("Error creating tensor for point cloud positions")

        self.last_seen = stamp
        self.first_seen = stamp
        self.is_confirmed = False
        self.centroids = (
            self.pcd.point.positions.cpu().numpy().mean(axis=0).reshape(1, 3)
        )

        r = self.pcd.cpu().to_legacy().get_rotation_matrix_from_xyz((0, 0, 0))
        self.quaternions = np.array([R.from_matrix(r).as_quat()])
        self.compute()

    def to_msg(self):
        """Returns the object in the format of the Object message."""
        obj = Object()
        obj.header.stamp = self.last_seen
        obj.id = self.id
        obj.points = self.pcd.point.positions.cpu().numpy().flatten().tolist()
        obj.label = self.label
        obj.score = self.score
        obj.bbox = self.bbox.flatten().tolist()
        return obj

    def update(self, other: "Obj"):
        """Updates the object's points and score and stores the historical state."""
        # saving other's centroid before registration
        other_centroid = other.pcd.point.positions.cpu().numpy().mean(axis=0)
        self.centroids = np.concatenate((self.centroids, [other_centroid]), axis=0)
        #
        ## saving other's quaternion before registration
        r = other.pcd.cpu().to_legacy().get_rotation_matrix_from_xyz((0, 0, 0))
        self.quaternions = np.concatenate(
            (self.quaternions, [R.from_matrix(r).as_quat()]), axis=0
        )
        #
        self.last_seen = max(self.last_seen, other.last_seen)
        self.is_confirmed = self.is_confirmed or other.is_confirmed
        # Euristics: an object is confirmed if it has been seen multiple times
        if not self.is_confirmed:
            self.is_confirmed = self.last_seen - self.first_seen >= rospy.Duration(
                TIME_TO_BE_CONFIRMED
            )

        # Use more sophisticated logic for updating point cloud data
        if self.label == other.label and self.label in MOVING_CLASSES:
            self.release_resources()
            self.pcd = other.pcd
        else:
            if self.first_seen < other.first_seen:
                target = self
                source = other
            else:
                target = other
                source = self

            source, res = self.register_pointcloud(source, target)
            self.pcd.point.positions = o3d.core.concatenate(
                (source.pcd.point.positions, target.pcd.point.positions), axis=0
            )

        self.label = self.label if self.score > other.score else other.label
        self.score = max(self.score, other.score)

        # indices = np.where(self.centroids != np.zeros(3))[0]
        # weights = exponential_weights(len(indices), decay_rate=0.1)
        # centroid = np.average(self.centroids[indices], axis=0, weights=weights)
        # quaternion = weightedAverageQuaternions(self.quaternions[indices], weights)
        # rot_matrix = R.from_quat(quaternion).as_matrix()
        # self.pcd.translate(centroid, relative=False)
        # self.pcd.rotate(rot_matrix, center=np.zeros(3))

        self.compute()

    def register_pointcloud(self, source, target: "Obj", min_size=100):
        """Registers another point cloud to this one and returns the transformed point cloud."""
        source_pcd = source.pcd
        target_pcd = target.pcd

        if (
            len(source_pcd.point.positions) < min_size
            or len(target_pcd.point.positions) < min_size
        ):
            # It does not make sense to register two small point clouds, let's build them up first
            return source, False

        # Perform ICP registration on GPU
        criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30
        )

        registration_result = o3d.t.pipelines.registration.icp(
            source_pcd,
            target_pcd,
            0.5,
            np.eye(4),
            o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        if registration_result.fitness < 0.3:
            rospy.logwarn("ICP registration failed, skipping registration")
            return source, False

        transformation = registration_result.transformation
        source_pcd.transform(transformation)
        source.pcd = source_pcd
        return source, True

    def compute(self):
        """Computes bounding box and centroid for the point cloud."""
        self.downsample()
        if len(self.pcd.point.positions) > 40:

            self.pcd.estimate_normals(max_nn=30, radius=0.1)
            self.pcd, _ = self.pcd.remove_radius_outliers(10, VOXEL_SIZE * 2)

            if len(self.pcd.point.positions) > 30:
                clean, _ = self.pcd.remove_statistical_outliers(10, 1.0)
            else:
                clean = self.pcd

            self.bbox = self.compute_minimum_oriented_box(clean)
            self.centroid = self.pcd.point.positions.cpu().numpy().mean(axis=0)
            return
        else:
            self.bbox = np.zeros((8, 3))
            self.centroid = np.zeros(3)
        
           

    def downsample(self):
        """Downsamples the point cloud."""
        self.pcd = self.pcd.voxel_down_sample(voxel_size=np.float32(VOXEL_SIZE))

    def to_cpu(self):
        """Moves the point cloud to CPU."""
        self.device = o3d.core.Device("CPU:0")
        self.pcd = self.pcd.to(self.device)

    def to_gpu(self):
        """Moves the point cloud to GPU."""
        self.device = device
        self.pcd = self.pcd.to(self.device)

    def __repr__(self):
        return f"Obj(id={self.id}, label={self.label}, score={self.score}, centroid={self.centroid}, last_seen={self.last_seen})"

    def compute_minimum_oriented_box(self, pcd):
        points = pcd.point.positions.cpu().numpy()
        box = np.array(oriented_bounding_box_numpy(points))
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(box))
        return np.asarray(obb.get_box_points())

    def compute_z_oriented_bounding_box(self, pcd):
        temp = pcd.cpu().to_legacy()

        points = np.asarray(temp.points)

        mean = np.mean(np.asarray(points), axis=0)

        # Compute PCA on the XY components only
        xy_points = np.asarray(points)[:, :2] - mean[:2]
        cov_matrix = np.dot(xy_points.T, xy_points) / len(xy_points)
        eigvals, eigvecs = np.linalg.eig(cov_matrix)

        # Align primary component with the X-axis
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        R = temp.get_rotation_matrix_from_xyz((0, 0, -angle))
        temp.rotate(R, center=mean)

        # Compute the axis-aligned bounding box of the rotated point cloud
        aabb = temp.get_axis_aligned_bounding_box()
        # convert aabb to oriented bbox
        aabb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(
            aabb
        )

        # Apply inverse rotation to the bounding box
        aabb.rotate(R.T, center=mean)

        return np.asarray(aabb.get_box_points())

    def compute_oriented_bounding_box(self, pcd):
        bbox = self.pcd.get_oriented_bounding_box()
        return bbox.get_box_points().cpu().numpy()

    def dynamic_voxel_downsample(self, factor=0.15):
        # Compute the bounding box
        bbox = self.pcd.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()

        # Calculate voxel size as a fraction of the smallest bounding box dimension
        voxel_size = min(bbox_extent) * factor

        # Perform voxel downsampling
        self.pcd = self.pcd.voxel_down_sample(voxel_size)
        return voxel_size

    def release_resources(self):
        """Safely releases all resources used by the object, including GPU memory."""
        # Ensure all tensors are deallocated
        self.pcd.clear()


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
            rospy.loginfo(
                "CUDA is available in Open3D, PointCloud operations will be managed on GPU"
            )
        else:
            rospy.loginfo(
                "CUDA is not available in Open3D, PointCloud operations will be managed on CPU"
            )

    # @time_it
    def add_object(self, obj):
        """Adds a new object to the world and updates the KDTree."""

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
        print("Removing objects: ", [self.objects[obj_id].label for obj_id in obj_ids])
        for obj_id in obj_ids:
            if obj_id in self.objects:
                self.objects[obj_id].release_resources()
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

        print("Is in world:", self.objects[obj.id].label)
        try:
            self.objects[obj.id].update(old_obj)
        except Exception as e:
            rospy.logerr(f"Error in overriding object: {e}")
            return
        self.remove_objects([old_id])
        rospy.loginfo(
            f"Overriding object {old_obj.id}:{old_obj.label} with {obj.id}:{obj.label}"
        )

    def get_objects(self):
        """Returns all objects in the world."""
        msg = ObjectList()
        msg.header.stamp = rospy.Time.now()

        msg.objects = list(self.objects.values())
        msg.objects = [obj.to_msg() for obj in msg.objects if obj.is_confirmed]

        return msg

    def query_by_distance(self, point, threshold):
        """Queries objects within a certain distance threshold."""
        if self.kdtree is not None:
            indexes = self.kdtree.query_ball_point(point, r=threshold)
            return [
                self.objects[obj_id]
                for obj_id in np.array(list(self.objects.keys()))[indexes]
            ]
        return []

    def query_close_objects_service(self, point, threshold):
        """Queries objects within a certain distance threshold."""
        msg = ObjectList()
        msg.header.stamp = rospy.Time.now()  # stamp of the query
        if self.kdtree is not None:
            indexes = self.kdtree.query_ball_point(point, r=threshold)
            objects = [
                self.objects[obj_id].to_msg()
                for obj_id in np.array(list(self.objects.keys()))[indexes]
            ]
            msg.objects = objects

        return msg

    def _rebuild_kdtree(self):

        if len(self.points_list) > 0:
            try:
                self.kdtree = KDTree(np.array(self.points_list))
            except:
                rospy.logerr("Error in rebuilding KDTree")

    # @time_it
    def get_world_id(self, obj: Obj, distance_thr=1, iou_thr=0.1):
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
                and abs(obj.last_seen - close_obj.last_seen).to_sec() > 1.0
            ):
                if compute_3d_iou(obj.bbox, close_obj.bbox) > iou_thr:
                    return close_obj.id

            # If the object was detected but misclassified
            if (
                obj.id != close_obj.id
                and obj.label != close_obj.label
                and abs(obj.last_seen - close_obj.last_seen).to_sec() < 1.0
            ):
                if compute_3d_iou(obj.bbox, close_obj.bbox) > 0.8:
                    return close_obj.id

        return obj.id

    # @time_it
    def manage_object(self, obj: Obj):
        std = np.std(obj.pcd.point.positions.cpu().numpy(), axis=0).max()
        if np.sum(obj.bbox) == 0 or std > STD_THR:
            return
        try:
            if obj.id in self.objects:
                self.update_object(obj)
                # taking the updated object
                obj = self.objects[obj.id]
            else:
                self.add_object(obj)

            world_id = self.get_world_id(obj, iou_thr=IOU_THR)

            if world_id != obj.id:
                self.override_object(world_id, obj)

            assert obj.id in self.objects
            if world_id != obj.id:
                assert world_id not in self.objects

        except Exception as e:
            rospy.logerr(f"Error in managing object: {e}")
            if obj.id in self.objects:
                self.remove_objects([obj.id])

    def clean_up(self):
        """
        Removes moving object that are not currently tracked or objects which Pointcloud has a std greater than a thr.
        """
        try:
            to_remove = {}
            for obj in self.objects.values():
                if (
                    obj.label in MOVING_CLASSES
                    and obj.last_seen.to_sec()
                    < rospy.Time.now().to_sec() - EXPIRY_TIME_MOVING_OBJECTS
                ):
                    to_remove[obj.id] = "Expired"

                if "positions" in obj.pcd.point:
                    std = np.std(obj.pcd.point.positions.cpu().numpy(), axis=0).max()
                    if std > STD_THR:
                        to_remove[obj.id] = (
                            f"->{obj.label}: STD above Thr: {std}> {STD_THR}"
                        )

            if len(to_remove) > 0:
                rospy.loginfo(f"Removing objects: {to_remove}")
                self.remove_objects(list(to_remove.keys()))
        except Exception as e:
            rospy.logerr(f"Error in cleaning up: {e}")

    def get_kdtree_centroids(self):
        return self.points_list
