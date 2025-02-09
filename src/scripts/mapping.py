#! /usr/bin/env python3
# /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ROS packages
import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray
from s_map.msg import Detection, Object, ObjectList
from s_map.srv import ManageObject, RemoveObjects, GetAllObjects, GetAllObjectsResponse, QueryObjects, QueryObjectsResponse, CleanUp
import rospkg
import copy
from geometry_msgs.msg import Point
import cv2

# Local modules
from utils import (
    create_marker_array,
    create_delete_marker,
    create_pointcloud_message,
    time_it,
    create_marker_point,
)
import supervision as sv
import numpy as np
from geometric_transformations import CameraPoseEstimator, TransformHelper
from pose_reliability import ReliabilityEvaluator


# Topic constants
RESULT_TOPIC = "/s_map/detection/results"
CAMERA_INFO_TOPIC = None
SCAN_TOPIC = "/scan"
MARKERS_TOPIC = "/s_map/objects"
PC_TOPIC = "/s_map/pointcloud"
WORLD_FRAME = None
# Frame constants
PKG_PATH = rospkg.RosPack().get_path("s_map")

MAX_DEPTH = 6.0
MIN_DEPTH = 0.8
EXPIRY_TIME = 10.0
QUEUE_SIZE = 5


class Mapper(object):
    def __init__(self):
        global WORLD_FRAME
        rospy.init_node("mapping", anonymous=True)
        self.pose_estimator_dict = {}
        self.transformer = TransformHelper()
        self.pose_reliability_evaluator = {}
        self.cv_bridge = CvBridge()
        self.init_params()
        self.init_subscribers()
        self.init_publishers()
        self.init_services()
        rospy.loginfo("Mapping node initialized")
        self.tracked_id = None
        rospy.Timer(rospy.Duration(0.3), self.clean_up)
    
    def init_services(self):
        rospy.wait_for_service("manage_object")
        rospy.wait_for_service("remove_objects")
        rospy.wait_for_service("get_all_objects")
        rospy.wait_for_service("query_objects")
        rospy.wait_for_service("clean_up")
        self.world_manager_client = rospy.ServiceProxy("manage_object", ManageObject)
        self.world_remove_client = rospy.ServiceProxy("remove_objects", RemoveObjects)
        self.world_all_objects_client = rospy.ServiceProxy("get_all_objects", GetAllObjects)
        self.world_query_client = rospy.ServiceProxy("query_objects", QueryObjects)
        self.world_clean_up_client = rospy.ServiceProxy("clean_up", CleanUp)
        rospy.loginfo("World Manager Services initialized for Mapping node")

    def init_params(self):
        global CAMERA_INFO_TOPIC, WORLD_FRAME
        CAMERA_INFO_TOPIC = rospy.get_param("~camera_info_topic", None)
        WORLD_FRAME = rospy.get_param("~world_frame", None)
        if not CAMERA_INFO_TOPIC or not WORLD_FRAME:
            rospy.logerr("Camera info topic or world frame not specified")
            rospy.signal_shutdown("Camera info topic or world frame not specified")

    def init_subscribers(self):
        self.result_subscriber = rospy.Subscriber(
            RESULT_TOPIC, Detection, self.process_data, queue_size=2*QUEUE_SIZE
            )

    def init_publishers(self):
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, MarkerArray, queue_size=QUEUE_SIZE)
        self.pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=QUEUE_SIZE)

    # @time_it
    def preprocess_msg(self, msg: Detection):
        header = msg.header
        labels = np.array(msg.labels)
        n = len(labels)
        boxes = np.array(msg.boxes).reshape(n, 4)
        scores = np.array(msg.scores)
        ids = np.array(msg.ids)
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg.depth, "16UC1")

        masks = self.cv_bridge.imgmsg_to_cv2(msg.masks)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=2)
        masks = np.moveaxis(masks, 2, 0)
        masks = masks.astype(np.uint8)

        if masks.shape[0] != n:
            masks = [None] * n

        return header, boxes, labels, scores, ids, masks, depth_image, msg.camera_name

    def is_reliable(self, header):
        if header.frame_id not in self.pose_reliability_evaluator:
            self.pose_reliability_evaluator[header.frame_id] = ReliabilityEvaluator(
                header.frame_id, WORLD_FRAME
            )
        return self.pose_reliability_evaluator[header.frame_id].evaluate(header.stamp)

    def get_pose_estimator(self, camera_name):
        if camera_name not in self.pose_estimator_dict.keys():
            self.pose_estimator_dict[camera_name] = CameraPoseEstimator(
                CAMERA_INFO_TOPIC
            )
        return self.pose_estimator_dict[camera_name]

    # @time_it
    def process_data(self, detection):
        if not self.is_reliable(detection.header):
            return

        header, boxes, labels, scores, ids, masks, depth_image, camera_name = (
            self.preprocess_msg(detection)
        )
        pose_estimator = self.get_pose_estimator(camera_name)
        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            obj = self.compute_object(
                id, box, depth_image, mask, label, score, header, pose_estimator
            )
            if obj is None:
                continue

            self.manage_object(obj)

    def manage_object(self, obj):
        res = self.world_manager_client(obj)

    # @time_it
    def clean_up(self, event):
        try:
            res = self.world_clean_up_client()
            #rospy.logwarn(f"Clean up: {res}")
        except Exception as e:
            rospy.logerr("Clean up service exception: : " + str(e))

    # @time_it
    def compute_object(
        self, id, box, depth_image, mask, label, score, header, pose_estimator
    ):
        """
        Computes the object information in the world frame
        Args:
            id (int): The ID of the object.
            box (tuple): The bounding box coordinates (xmin, ymin, xmax, ymax).
            depth_image (numpy.ndarray): The depth image.
            mask (numpy.ndarray): The mask indicating the object region.
            label (str): The label of the object.
            score (float): The score of the object.
            stamp (float): The timestamp of the object.

        Returns:
            Obj: An instance of the Obj class representing the computed object.
                Returns None if the computation fails.
        """
        pc = self.compute_pointcloud(depth_image, mask)
        if len(pc) < 10:
            return None

        pc_camera_frame = pose_estimator.multiple_pixels_to_3d(pc)
        pc_world_frame = self.transformer.fast_transform(
            header.frame_id, WORLD_FRAME, pc_camera_frame, header.stamp
        )

        if pc_world_frame is None:
            return None

        obj = Object()
        obj.header = copy.deepcopy(header)
        obj.header.frame_id = WORLD_FRAME
        obj.id = int(id)
        obj.points = pc_world_frame.astype(np.float32).flatten().tolist()
        obj.label = label
        obj.score = score
        return obj

    # @time_it
    def compute_pointcloud(self, depth_image, mask):
        pc = depth_image * mask
        (ys, xs) = np.argwhere(pc).T

        # remove from the pointcloud the points that are too close or too far from the camera
        zs = pc[ys, xs] / 1000
        mask = np.logical_and(zs > MIN_DEPTH, zs < MAX_DEPTH)
        ys = ys[mask]
        xs = xs[mask]
        zs = zs[mask]

        pointcloud = np.array([xs, ys, zs]).T

        return pointcloud

    """
    def plot_centroids(self, event):
        objects = self.world_all_objects_client().objects.objects
        centroids = []
        for obj in objects:
            points = np.array(obj.points).reshape(-1, 3)
            centroid = np.mean(points, axis=0)
            centroids.append(centroid)

        marker = create_marker_point(centroids, rospy.Time.now(), WORLD_FRAME)
        self.marker_pub.publish(marker)
        """


if __name__ == "__main__":
    mapper = Mapper()
    rospy.spin()
