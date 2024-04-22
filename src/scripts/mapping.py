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
from utils import create_marker_vertices, get_vercitces, bbox_iou


# others
# import cv2
import supervision as sv
import numpy as np
import time
from ctypes import *
from geometric_transformations import CameraPoseEstimator, TransformHelper
import math
import cv2
from collections import deque
from world import Obj, World
from pose_reliability import ReliabilityEvaluator

RESULT_TOPIC = "/s_map/detection/results"
DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
RGB_TOPIC = "/realsense/rgb/image_raw"
CAMERA_INFO_TOPIC = "/realsense/aligned_depth_to_color/camera_info"
SCAN_TOPIC = "/scan"
MARKERS_TOPIC = "/s_map/objects"

WORLD_FRAME = "world"
CAMERA_FRAME = "realsense_rgb_optical_frame"
RGB_FRAME = "realsense_rgb_frame"


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
        "pose_reliability_evaluator",
    ]
    cv_bridge: CvBridge
    result_subscriber: Subscriber
    laser_subscriber: Subscriber
    depth_subscriber: Subscriber
    transformer: TransformHelper
    pose_estimator: CameraPoseEstimator
    marker_pub: rospy.Publisher
    world: World
    pose_reliability_evaluator: ReliabilityEvaluator

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
        self.pose_reliability_evaluator = ReliabilityEvaluator(
            CAMERA_FRAME, WORLD_FRAME
        )
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

        # boxes = []
        # for mask in masks:
        #    obj_presence_idx = np.argwhere(mask)
        #    if len(obj_presence_idx) != 0:
        #        ymin, xmin = np.min(obj_presence_idx, axis=0)
        #        ymax, xmax = np.max(obj_presence_idx, axis=0)
        #    else:
        #        ymin, xmin, ymax, xmax = 0, 0, 0, 0
        #    boxes.append([xmin, ymin, xmax, ymax])

        return header, boxes, labels, scores, ids, masks

    def still_exist(self, boxes, labels, header, width=848, height=480):
        # start = time.time()
        for id, object in self.world.objects.items():
            points = object["actual"].points
            points = self.transformer.transform_coordinates(
                WORLD_FRAME, CAMERA_FRAME, points, header.stamp
            )
            if np.min(points[:, 2]) < 1:
                # if the object is behind the camera
                continue

            pixels = np.array(
                [self.pose_estimator.d3_to_pixel(*point) for point in points]
            )

            xcoords = sorted(pixels[:, 0])
            ycoords = sorted(pixels[:, 1])
            xmin = np.max(xcoords[:4])
            xmax = np.min(xcoords[4:])
            ymin = np.max(ycoords[:4])
            ymax = np.min(ycoords[4:])

            # xmin = np.min(pixels[:, 0])
            # xmax = np.max(pixels[:, 0])
            # ymin = np.min(pixels[:, 1])
            # ymax = np.max(pixels[:, 1])

            bbox = np.array(
                [xmin, ymin, xmax, ymax],
            )
            # Idea: check if the visisble portion of the bbox has an IoU > thr with the original bbox
            visible_bbox = np.array(
                [max(0, xmin), max(0, ymin), min(width, xmax), min(height, ymax)],
            )
            iou = bbox_iou(bbox, visible_bbox)
            found = False
            if iou > 0.7:
                rospy.loginfo(f"Object {str(id)} should be visible, %visibility: {iou}")
                for box, label in zip(boxes, labels):
                    iou = bbox_iou(box, visible_bbox)
                    if iou > 0.1 and label == object["actual"].label:
                        rospy.loginfo(
                            f"Object {str(id)} is still visible, pred label:{label}, actual label:{object['actual'].label}, %match: {iou}"
                        )
                        found = True
                        break

                if not found:
                    rospy.logwarn(f"Object {str(id)} is not visible anymore")

        # end = time.time()
        # print("Time to check if objects are still in the image plane: ", end - start)

    def mapping_callback(self, detection_msg: Detection, depth_msg: Image):
        # start = time.time()
        if not self.pose_reliability_evaluator.evaluate(detection_msg.header.stamp):
            #rospy.logwarn("Pose is not reliable")
            return
        header, boxes, labels, scores, ids, masks = self.preprocess_msg(detection_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        self.still_exist(boxes, labels, header)
        # print(labels)
        # print(boxes)

        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            # for more precise measurement, we extract min max coordinates from mask indices
            xmin, ymin, xmax, ymax = box

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
            vertices_world_frame = self.transformer.transform_coordinates(
                CAMERA_FRAME, WORLD_FRAME, vertices_camera_frame, header.stamp
            )

            if vertices_world_frame is None:
                rospy.logwarn(
                    "[Mapping] One of the vertices is None even if depth was found"
                )
                continue

            object = Obj(vertices_world_frame, label, score)
            # overwriting id in case it has changed (e.g. object was already registered in the world)
            self.world.register_object(id, object)
            object, id = self.world.get_object(id)

            # creating marker message for rviz visualization
            marker = create_marker_vertices(
                object.points, object.label, id, header.stamp, WORLD_FRAME
            )

            if marker is not None:
                self.marker_pub.publish(marker)
        # end = time.time()
        # print("Time to process detection: ", end - start)


if __name__ == "__main__":
    Mapper()
