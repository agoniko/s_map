#! /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ros packages
import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose

# others
# import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from ctypes import *
from geometric_transformations import Transformer, CameraPoseEstimator
import math


class Node(object):
    def __init__(self):
        rospy.init_node("detection")
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        rospy.loginfo(f"Using {self.device} device for detection")

        # find models path inside package with rospy
        model_path = rospkg.RosPack().get_path("s_map") + "/models/yolov8n-seg.pt"
        rospy.loginfo(f"Model path: {model_path}")
        self.detector = YOLO(model_path)
        self.detectot = self.detector.to(self.device)

        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxCornerAnnotator()

        # pub_sub for detection
        rospy.Subscriber(
            "/realsense/rgb/image_raw", Image, self.detection_callback, queue_size=1
        )
        self.pub = rospy.Publisher(
            "/s_map/detection/image_annotated", Image, queue_size=1
        )

        # pub_sub for scan
        rospy.Subscriber("/b_scan", LaserScan, self.scan_callback, queue_size=1)
        self.pub_scan = rospy.Publisher("/s_map/scan", LaserScan, queue_size=1)
        self.transformer = Transformer()

        # sub for robot orientation
        self.pose_sub = rospy.Subscriber(
            "/robot_pose", Pose, self.pose_callback, queue_size=1
        )
        self.robot_position = None
        self.robot_orientation = None

        rospy.spin()

    def pose_callback(self, msg):
        self.robot_orientation = msg.orientation
        self.robot_position = msg.position

    def detection_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = next(
            self.detector(
                frame, device=self.device, stream=True, conf=0.4, verbose=False
            )
        )
        for res in results:
            detections = sv.Detections.from_ultralytics(res)
            frame = self.mask_annotator.annotate(frame, detections)
            frame = self.label_annotator.annotate(frame, detections)
            frame = self.box_annotator.annotate(frame, detections)

        annotated_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.pub.publish(annotated_image_msg)

    def scan_callback(self, msg):
        pass


if __name__ == "__main__":
    node = Node()
