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


def create_marker(point_min, point_max, label, id, stamp) -> Marker:
    marker = Marker()
    marker.header.frame_id = (
        WORLD_FRAME  # Specify the frame in which the point is defined
    )
    marker.id = id
    marker.header.stamp = stamp

    marker.ns = "my_namespace"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0

    marker.color.a = 1.0

    marker.pose.position.x = (point_min[0] + point_max[0]) / 2
    marker.pose.position.y = (point_min[1] + point_max[1]) / 2
    marker.pose.position.z = (point_min[2] + point_max[2]) / 2
    marker.scale.x = point_max[0] - point_min[0]
    marker.scale.y = point_max[1] - point_min[1]
    marker.scale.z = point_max[2] - point_min[2]

    # quaternion = np.random.rand(4)
    quaternion = calculate_orientation_quaternion(*point_min, *point_max)
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    if label.lower() == "person":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    elif label.lower() == "chair":
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "laptop":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    elif label.lower() == "dining table":
        # yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "tv":
        # aqua
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
    else:
        return None

    return marker


def create_marker_vertices(vertices, label, id, stamp) -> Marker:
    marker = Marker()
    marker.header.frame_id = (
        WORLD_FRAME  # Specify the frame in which the point is defined
    )
    marker.id = id
    marker.header.stamp = stamp

    marker.ns = "my_namespace"
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.color.a = 1.0
    marker.scale.x = 0.05

    # create the lines of the bounding box
    # bottom face
    marker.points.append(Point(*vertices[0]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[0]))

    # top face
    marker.points.append(Point(*vertices[4]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[7]))
    marker.points.append(Point(*vertices[7]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[4]))

    # vertical lines
    marker.points.append(Point(*vertices[0]))
    marker.points.append(Point(*vertices[4]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[7]))

    if label.lower() == "person":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    elif label.lower() == "chair":
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "laptop":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    elif label.lower() == "dining table":
        # yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "tv":
        # aqua
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
    else:
        return None

    return marker


def get_vercitces(point_min, point_max):
    vertices = [
        #     X             Y             Z
        [point_min[0], point_min[1], point_min[2]],
        [point_max[0], point_min[1], point_min[2]],
        [point_min[0], point_max[1], point_min[2]],
        [point_max[0], point_max[1], point_min[2]],
        [point_min[0], point_min[1], point_max[2]],
        [point_max[0], point_min[1], point_max[2]],
        [point_min[0], point_max[1], point_max[2]],
        [point_max[0], point_max[1], point_max[2]],
    ]
    return vertices


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

        self.transformer = TransformHelper()
        self.pose_estimator = CameraPoseEstimator(CAMERA_INFO_TOPIC)
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, Marker, queue_size=1)

        # stores id->List[Point, label, confidence_score], so that we can update the position of the object in the map
        self.world = World()
        rospy.spin()

    def preprocess_msg(self, msg: Detection):
        header = msg.header
        labels = np.array(msg.labels)
        n = len(labels)
        boxes = np.array(msg.boxes).reshape(n, 4)
        scores = np.array(msg.scores)
        ids = np.array(msg.ids)

        masks = self.cv_bridge.imgmsg_to_cv2(msg.masks)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=2)
        masks = np.moveaxis(masks, 2, 0)
        masks = masks.astype(np.uint8)

        if masks.shape[0] != n:
            masks = [None] * n

        return header, boxes, labels, scores, ids, masks

    def mapping_callback(self, detection_msg: Detection, depth_msg: Image):
        header, boxes, labels, scores, ids, masks = self.preprocess_msg(detection_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            xmin, ymin, xmax, ymax = box  # already scaled to the image size

            # TODO: check why sometimes mask number is different from boxes number,
            # For now if mask is none we extract depth from bbox
            if mask is not None:
                filtered_depth = depth_image * mask
                values = filtered_depth[filtered_depth != 0]
            else:
                rospy.logwarn(
                    "[Mapping] Mask is None, extracting depth from bounding box"
                )
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

            # Extracting the 3d bounding box in the world frame, we cannot use only min max points because we lose orientation
            vertices_camera_frame = get_vercitces(point_min, point_max)
            vertices_world_frame = [
                self.transformer.lookup_transform_and_transform_coordinates(
                    CAMERA_FRAME, WORLD_FRAME, vertex, header.stamp
                )
                for vertex in vertices_camera_frame
            ]
            vertices_world_frame = np.array(vertices_world_frame)

            if None in vertices_world_frame:
                rospy.logwarn(
                    "[Mapping] One of the vertices is None even if depth was found"
                )
                continue

            object = Obj(vertices_world_frame, label, score)
            # overwriting id in case it has changed (e.g. object was already registered in the world)
            self.world.register_object(id, object)
            object, id = self.world.get_object(id)

            marker = create_marker_vertices(
                object.points, object.label, id, header.stamp
            )

            if marker is not None:
                self.marker_pub.publish(marker)


if __name__ == "__main__":
    Mapper()
