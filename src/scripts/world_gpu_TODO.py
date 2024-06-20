import open3d as o3d
from scipy.spatial import KDTree
import numpy as np
from collections import deque
from utils import compute_3d_iou
from utils import time_it
import rospy
import copy

TIME_TO_BE_CONFIRMED = 0.0
MOVING_CLASSES = ["person"]
VOXEL_SIZE = 0.03
o3d.core.Device("CUDA:0")

class Obj:
    """Object class with historical data and spatial indexing using KDTree."""

    __slots__ = ["pcd", "label", "score", "bbox", "id", "centroid", "last_seen", "first_seen", "is_confirmed", "device"]

    def __init__(self, id, points, label, score, stamp):
        self.id = id
        self.label = label
        self.score = score
        self.device = o3d.core.Device("CPU:0")
        self.pcd = o3d.t.geometry.PointCloud(device = self.device)
        self.pcd.point.positions = o3d.core.Tensor(points, o3d.core.float32, self.device)
        self.last_seen = stamp
        self.first_seen = stamp
        self.is_confirmed = False
        self.compute()

    def update(self, other: "Obj"):
        """Updates the object's points and score and stores the historical state."""
        try:
            other = self.register_pointcloud_to_self(other)
        except Exception as e:
            rospy.logwarn(f"Error in registering point clouds: {e}")
            return

        self.last_seen = max(self.last_seen, other.last_seen)

        #Euristics: an object is confirmed if it has been seen multiple times
        if not self.is_confirmed:
            self.is_confirmed = self.last_seen - self.first_seen >= rospy.Duration(TIME_TO_BE_CONFIRMED)
        
        # Use more sophisticated logic for updating point cloud data
        if self.label == other.label and self.label in MOVING_CLASSES:
            self.pcd.point.positions = other.pcd.point.positions
        else:
            self.pcd.point.positions = o3d.core.concatenate((self.pcd.point.positions, other.pcd.point.positions), axis=0)

        self.compute()
        print(self.bbox, self.centroid)
        self.label = other.label if other.score > self.score else self.label
        self.score = max(self.score, other.score)

        
    def register_pointcloud_to_self(self, other: "Obj", threshold=0.01):
        """Aligns other point cloud to self using ICP registration."""
        try:
            reg_p2p = o3d.pipelines.registration.registration_icp(
                other.pcd.to_legacy(), self.pcd.to_legacy(), threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            """reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                                                                                            other.pcd.to_legacy(),
                                                                                            self.pcd.to_legacy(),
                                                                                            o3d.pipelines.registration.compute_fpfh_feature(other.pcd.to_legacy(), o3d.geometry.KDTreeSearchParamHybrid(radius= 2* VOXEL_SIZE, max_nn=30)),
                                                                                            o3d.pipelines.registration.compute_fpfh_feature(self.pcd.to_legacy(), o3d.geometry.KDTreeSearchParamHybrid(radius= 2* VOXEL_SIZE, max_nn=30)),
                                                                                            True,
                                                                                            threshold,
                                                                                            )
            """
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

    def compute(self):
        #print(self.pcd)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        self.pcd, _ = self.pcd.remove_radius_outliers(nb_points=50, search_radius=0.5)
        self.pcd.estimate_normals(max_nn=30, radius = 1.4 * VOXEL_SIZE) #Recommended by documentation
        #clean, _ = self.pcd.remove_radius_outlier(nb_points=10, radius=0.5)
        clean, _ = self.pcd.remove_statistical_outliers(nb_neighbors=50, std_ratio=0.1)

        if len(clean.point.positions.cpu().numpy()) <= 10:
            self.bbox = np.zeros((8, 3))
            self.centroid = np.zeros(3)
            return

        self.bbox = self.compute_z_oriented_bounding_box(clean)
        print(self.bbox)
        self.centroid = np.mean(self.bbox, axis=0)


class World:
    """World class that uses a KDTree for efficient spatial management of objects."""
    def __init__(self):
        self.objects = {}
        self.kdtree = None
        self.points_list = []
        self.id2index = {}
        self.index2id = {}
        if o3d.core.cuda.is_available():
            print("CUDA is available in Open3D.")
        else:
            print("CUDA is not available in Open3D.")

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

    
    #@time_it
    def remove_objects(self, obj_ids):
        """Removes an object from the world."""
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
        rospy.loginfo(f"Overriding object {old_id}:{old_obj.label} with {obj.id}:{obj.label}")
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
            self.kdtree = KDTree(np.array(self.points_list))

    # @time_it
    def get_world_id(self, obj: Obj, distance_thr=1, iou_thr=0.0):
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
                # print(f"Distance: {distance} between {obj.id}:{obj.label} and {close_obj.id}:{close_obj.label}")
                if obj.label == "chair":
                    print(f"CHAIRS: {obj.id} and {close_obj.id}")
                    print(compute_3d_iou(obj.bbox, close_obj.bbox))
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
    
    def remove_old_moving_objects(self):
        to_remove = []
        for obj in self.objects.values():
            if obj.label in MOVING_CLASSES and obj.last_seen.to_sec() < rospy.Time.now().to_sec() - 1.0:
                to_remove.append(obj.id)
        
        self.remove_objects(to_remove)
