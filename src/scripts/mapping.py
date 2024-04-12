#! /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ros packages
import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from s_map.msg import Detection
from message_filters import TimeSynchronizer, Subscriber
from collections import Counter
from tf.transformations import quaternion_from_euler


# others
# import cv2
import supervision as sv
import numpy as np
import time
from ctypes import *
from geometric_transformations import Transformer, CameraPoseEstimator, TransformHelper
import math
import cv2
from collections import deque
from world import Obj, World

RESULT_TOPIC = "/s_map/detection/results"
DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
RGB_TOPIC = "/realsense/rgb/image_raw"
CAMERA_INFO_TOPIC = "/realsense/aligned_depth_to_color/camera_info"
SCAN_TOPIC = "/scan"
MARKERS_TOPIC = "/s_map/objects"

WORLD_FRAME = "world"
CAMERA_FRAME = "realsense_rgb_optical_frame"


def create_marker() -> Marker:
    marker_msg = Marker()
    marker_msg.header.frame_id = (
        WORLD_FRAME  # Specify the frame in which the point is defined
    )
    marker_msg.ns = "my_namespace"
    marker_msg.type = Marker.CUBE
    marker_msg.action = Marker.ADD
    marker_msg.pose.orientation.w = 1.0
    marker_msg.scale.x = 0.2  # Size of the points
    marker_msg.scale.y = 0.2
    marker_msg.scale.z = 0.2
    marker_msg.color.a = 1.0  # Alpha (transparency)
    marker_msg.color.r = 0.0
    marker_msg.color.g = 0.0
    marker_msg.color.b = 0.0
    return marker_msg


def create_point_msg(x, y, z):
    point = Point()
    point.x = x
    point.y = y
    point.z = z
    return point


class Mapper(object):
    __slots__ = [
        "cv_bridge",
        "result_subscriber",
        "laser_subscriber",
        "depth_subscriber",
        "transformer",
        "pose_estimator",
        "syncronizer",
        "marker_pub",
        "world",
    ]
    cv_bridge: CvBridge
    result_subscriber: Subscriber
    laser_subscriber: Subscriber
    depth_subscriber: Subscriber
    transformer: TransformHelper
    pose_estimator: CameraPoseEstimator
    marker_pub: rospy.Publisher
    world: World

    def __init__(self):
        """
        Initializes a ROS node for computing semantic map
        """
        rospy.init_node("mapping")
        self.cv_bridge = CvBridge()

        # synchronized subscriber for detection and depth
        self.result_subscriber = Subscriber(RESULT_TOPIC, Detection)
        self.depth_subscriber = Subscriber(DEPTH_TOPIC, Image)
        self.syncronizer = TimeSynchronizer(
            [self.result_subscriber, self.depth_subscriber], 1000
        )
        self.syncronizer.registerCallback(self.mapping_callback)

        # pub_sub for scan
        # self.laser_subscriber = rospy.Subscriber(
        #    "/scan", LaserScan, self.scan_callback, queue_size=1
        # )

        self.transformer = TransformHelper()
        self.pose_estimator = CameraPoseEstimator(CAMERA_INFO_TOPIC)
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, Marker, queue_size=1)

        # stores id->List[Point, label, confidence_score], so that we can update the position of the object in the map
        self.world = World()
        rospy.spin()

    def mapping_callback(self, detection_msg: Detection, depth_msg: Image):
        header = detection_msg.header
        labels = np.array(detection_msg.labels)
        n = len(labels)
        boxes = np.array(detection_msg.boxes).reshape(n, 4)
        scores = np.array(detection_msg.scores)
        ids = np.array(detection_msg.ids)
        # TODO: masks
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        marker_msg = create_marker()
        marker_msg.header.stamp = header.stamp

        for id, box, label, score in zip(ids, boxes, labels, scores):
            xmin, ymin, xmax, ymax = box

            # scaling coordinates to depth image size (it is the same of the rgb image size)
            xmin = int(xmin * depth_image.shape[1])
            xmax = int(xmax * depth_image.shape[1])
            ymin = int(ymin * depth_image.shape[0])
            ymax = int(ymax * depth_image.shape[0])

            # checking median depth value (in meters) inside the bounding box excluding zeros
            values = depth_image[ymin:ymax, xmin:xmax][
                depth_image[ymin:ymax, xmin:xmax] != 0
            ]
            if len(values) == 0:
                continue
            zmin, zmax = np.min(values) / 1000, np.max(values) / 1000  # depth in meters

            # those points refers to meters coordinate of a 3d bounding box w.r.t the camera frame
            point_min = [*self.pose_estimator.pixel_to_3d(xmin, ymin, zmin), zmin]
            point_max = [*self.pose_estimator.pixel_to_3d(xmax, ymax, zmax), zmax]

            point_min_world = (
                self.transformer.lookup_transform_and_transform_coordinates(
                    CAMERA_FRAME, WORLD_FRAME, point_min, header.stamp
                )
            )
            point_max_world = (
                self.transformer.lookup_transform_and_transform_coordinates(
                    CAMERA_FRAME, WORLD_FRAME, point_max, header.stamp
                )
            )

            if point_min_world is None or point_max_world is None:
                rospy.logwarn(f"[Mapping] Point is None even if depth was found")
                continue

            point_min_world = [
                point_min_world.point.x,
                point_min_world.point.y,
                point_min_world.point.z,
            ]
            point_max_world = [
                point_max_world.point.x,
                point_max_world.point.y,
                point_max_world.point.z,
            ]

            bbox_3d = np.array([point_min_world, point_max_world])
            object = Obj(bbox_3d, label, score)
            # overwriting id in case it has changed (e.g. object was already registered in the world)
            self.world.register_object(id, object)

            object, id = self.world.get_object(id)
            # centroid = np.median(object.points, axis=0)
            # label = object.label
            #
            # marker_msg.points.append(create_point_msg(*centroid))
            # marker_msg.id = id
            point_min = object.points[0]
            point_max = object.points[1]

            marker_msg.pose.position.x = (point_min[0] + point_max[0]) / 2
            marker_msg.pose.position.y = (point_min[1] + point_max[1]) / 2
            marker_msg.pose.position.z = (point_min[2] + point_max[2]) / 2
            marker_msg.scale.x = point_max[0] - point_min[0]
            marker_msg.scale.y = point_max[1] - point_min[1]
            marker_msg.scale.z = point_max[2] - point_min[2]
            marker_msg.pose.orientation.w = 1.0

            marker_msg.id = id

            # print(label)
            # print("Point min: ", point_min)
            # print("Point max: ", point_max)
            # print("______________________")

            if label.lower() == "person":
                marker_msg.color.r = 1.0
                marker_msg.color.g = 0.0
                marker_msg.color.b = 0.0
            elif label.lower() == "chair":
                marker_msg.color.r = 0.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 0.0
            elif label.lower() == "laptop":
                marker_msg.color.r = 0.0
                marker_msg.color.g = 0.0
                marker_msg.color.b = 1.0
            elif label.lower() == "dining table":
                # yellow
                marker_msg.color.r = 1.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 0.0
            elif label.lower() == "tv":
                # aqua
                marker_msg.color.r = 0.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 1.0
            else:
                continue
            self.marker_pub.publish(marker_msg)
            marker_msg.points = []
            # rospy.loginfo("[Mapping] Published object")


if __name__ == "__main__":
    Mapper()
