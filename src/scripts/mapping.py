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
SCAN_TOPIC = "/scan"


class Mapper(object):
    __slots__ = [
        "cv_bridge",
        "result_subscriber",
        "laser_subscriber",
        "depth_subscriber",
        "transformer",
        "syncronizer",
        "pub",
    ]
    cv_bridge: CvBridge
    result_subscriber: Subscriber
    laser_subscriber: Subscriber
    depth_subscriber: Subscriber
    transformer: TransformHelper
    pub: rospy.Publisher

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
            [self.result_subscriber, self.depth_subscriber], 1
        )
        self.syncronizer.registerCallback(self.mapping_callback)

        # pub_sub for scan
        # self.laser_subscriber = rospy.Subscriber(
        #    "/scan", LaserScan, self.scan_callback, queue_size=1
        # )

        self.transformer = TransformHelper()
        self.pub = rospy.Publisher("/s_map/test", Image, queue_size=1)
        rospy.spin()

    def mapping_callback(self, det_msg: Detection, depth_msg: Image):
        pass
        #print(det_msg.header.stamp)
        #print(depth_msg.header.stamp)
        # start = time.time()
        # header = msg.header
        # labels = np.array(msg.labels)
        # n = len(labels)
        # boxes = np.array(msg.boxes).reshape(n, 4)
        # masks = np.array(msg.masks).reshape(n, msg.height, msg.width)
        # depth_image = np.array(msg.depth_image).reshape(msg.height, msg.width).astype("uint16")


#
# depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, "16UC1")
# depth_msg.header = header
#
# self.pub.publish(depth_msg)
# rospy.loginfo(f"Mapping callback took {time.time() - start} seconds")


if __name__ == "__main__":
    Mapper()
