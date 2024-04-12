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
    marker_msg.type = Marker.POINTS
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
        "objects_dict",
    ]
    cv_bridge: CvBridge
    result_subscriber: Subscriber
    laser_subscriber: Subscriber
    depth_subscriber: Subscriber
    transformer: TransformHelper
    pose_estimator: CameraPoseEstimator
    marker_pub: rospy.Publisher
    objects_dict: dict

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
        self.objects_dict = dict()
        rospy.spin()

    def register_object(
        self, id: int, point: Point, label: str, confidence_score: float
    ):
        if id in self.objects_dict:
            self.objects_dict[id]["points"].append(point)
            self.objects_dict[id]["labels"].append(label)
            self.objects_dict[id]["scores"].append(confidence_score)
        else:
            self.objects_dict[id] = {
                "points": [point],
                "labels": [label],
                "scores": [confidence_score],
            }

    def best_label(self, labels, confidence_scores) -> int:
        """
        Returns the index of the best label as a function of both confidence scores and confidence score
        """
        # freqs = Counter(labels)
        # if len(freqs) == 1:
        #    return 0
        # scores = [np.mean(confidence_scores[label == labels]) for label in freqs.keys()]
        ## apply min_max normalization to both confidence scores and frequencies
        # min_conf, max_conf = min(scores), max(scores)
        # min_freq, max_freq = min(list(freqs.values())), max(list(freqs.values()))
        # norm_conf = [(conf - min_conf) / (max_conf - min_conf) for conf in scores]
        # norm_freq = [
        #    (freq - min_freq) / (max_freq - min_freq + 1e-08) for freq in freqs.values()
        # ]
        #
        ## compute the score as a weighted sum of the two normalized values
        # scores = [conf * freq for conf, freq in zip(norm_conf, norm_freq)]
        # return np.argmax(scores)
        return np.argmax(confidence_scores)

    def get_object(self, id: int) -> tuple[Point, str] | None:
        """returns the median Point and the best label or None if the id is not found"""
        if id in self.objects_dict:
            points = np.array(
                [
                    [point.x, point.y, point.z]
                    for point in self.objects_dict[id]["points"]
                ]
            )
            median = np.median(points, axis=0)
            labels = self.objects_dict[id]["labels"]
            confidence_scores = self.objects_dict[id]["scores"]
            idx = self.best_label(labels, confidence_scores)
            label = labels[idx]
            return Point(median[0], median[1], median[2]), label

        return None

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
            marker_msg.id = id
            x1, y1, x2, y2 = box

            # scaling coordinates to depth image size (it is the same of the rgb image size)
            x1 = int(x1 * depth_image.shape[1])
            x2 = int(x2 * depth_image.shape[1])
            y1 = int(y1 * depth_image.shape[0])
            y2 = int(y2 * depth_image.shape[0])

            # checking median depth value (in meters) inside the bounding box excluding zeros
            values = depth_image[y1:y2, x1:x2][depth_image[y1:y2, x1:x2] != 0]
            if len(values) == 0:
                continue
            depth = np.median(values) / 1000
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            x_3d, y_3d = self.pose_estimator.pixel_to_3d(x_center, y_center, depth)

            point_source = PointStamped()
            point_source.header.frame_id = CAMERA_FRAME
            point_source.point.x = x_3d
            point_source.point.y = y_3d
            point_source.point.z = depth

            transformed_point = (
                self.transformer.lookup_transform_and_transform_coordinates(
                    CAMERA_FRAME, WORLD_FRAME, point_source, header.stamp
                )
            )

            if transformed_point is None:
                rospy.logwarn(
                    f"[Mapping] Point is None even if depth was found\n{x_center, y_center, depth}"
                )
                continue

            point = transformed_point.point
            self.register_object(id, point, label, score)

            point, label = self.get_object(id)
            marker_msg.points.append(point)

            if label.lower() == "person":
                marker_msg.color.r = 1.0
            elif label.lower() == "chair":
                marker_msg.color.g = 1.0
            elif label.lower() == "laptop":
                marker_msg.color.b = 1.0
            elif label.lower() == "dining table":
                marker_msg.color.r = 1.0
                marker_msg.color.b = 1.0
            elif label.lower() == "tv":
                marker_msg.color.g = 1.0
                marker_msg.color.b = 1.0
            self.marker_pub.publish(marker_msg)
            marker_msg.points = []
            rospy.loginfo("[Mapping] Published object")


if __name__ == "__main__":
    Mapper()
