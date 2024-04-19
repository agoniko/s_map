#! /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ros packages
import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from s_map.msg import Detection
from message_filters import TimeSynchronizer, Subscriber
from collections import Counter
from tf.transformations import quaternion_from_euler
from utils import create_marker_vertices, get_vercitces


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
        self.world = World()

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

    def still_exist(self, boxes, header, width=848, height=480):
        for id, object in self.world.objects.items():
            points = object["actual"].points
            points = np.array(
                [
                    self.transformer.lookup_transform_and_transform_coordinates(
                        WORLD_FRAME, CAMERA_FRAME, point, header.stamp
                    )
                    for point in points
                ]
            )

            if np.min(points[:, 2]) < 1:
                # if the object is behind the camera
                continue

            pixels = np.array(
                [self.pose_estimator.d3_to_pixel(*point) for point in points]
            )

            x_pixels = sorted(pixels[:, 0])
            y_pixels = sorted(pixels[:, 1])

            xmin = np.max(x_pixels[:4])
            xmax = np.min(x_pixels[4:])
            ymin = np.max(y_pixels[:4])
            ymax = np.min(y_pixels[4:])

            bbox = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ]
            )
            # check if the object is still in the image
            if all(
                [
                    any(bbox.flatten() < 0)
                    or any(bbox[:, 0] > width)
                    or any(bbox[:, 1] > height)
                ]
            ):
                continue

            print(bbox)
            print(f"Object {str(id)} still exists")

    def mapping_callback(self, detection_msg: Detection, depth_msg: Image):
        header, boxes, labels, scores, ids, masks = self.preprocess_msg(detection_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        self.still_exist(boxes, header)

        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            # for more precise measurement, we extract min max coordinates from mask indices
            ymin, xmin = np.min(np.argwhere(mask), axis=0)
            ymax, xmax = np.max(np.argwhere(mask), axis=0)

            # overlapping depth and mask
            filtered_depth = depth_image * mask
            values = filtered_depth[filtered_depth != 0]

            if len(values) == 0:
                continue

            zmin, zmax = np.min(values) / 1000, np.max(values) / 1000  # depth in meters

            # those points refers to meters coordinate of a 3d bounding box w.r.t the camera frame
            point_min = [*self.pose_estimator.pixel_to_3d(xmin, ymin, zmin), zmin]
            point_max = [*self.pose_estimator.pixel_to_3d(xmax, ymax, zmax), zmax]

            # Extracting the 3d bounding box in the world frame, we cannot use only min max points because we lose orientation
            vertices_camera_frame = get_vercitces(point_min, point_max)
            vertices_world_frame = np.array(
                [
                    self.transformer.lookup_transform_and_transform_coordinates(
                        CAMERA_FRAME, WORLD_FRAME, vertex, header.stamp
                    )
                    for vertex in vertices_camera_frame
                ]
            )

            if None in vertices_world_frame:
                rospy.logwarn(
                    "[Mapping] One of the vertices is None even if depth was found"
                )
                continue

            object = Obj(vertices_world_frame, label, score)
            # overwriting id in case it has changed (e.g. object was already registered in the world)
            self.world.register_object(id, object)
            object, id = self.world.get_object(id)

            # creating marker message for rviz visualization
            marker = create_marker_vertices(object.points, object.label, id, header)

            if marker is not None:
                self.marker_pub.publish(marker)


if __name__ == "__main__":
    Mapper()
