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
import cv2
from collections import deque


class Node(object):
    def __init__(self):
        """
        Initializes a ROS node for object detection and laser scan processing.

        This class sets up the necessary ROS subscriptions and publishers for processing image data from the RealSense camera and laser scan data from the robot. It loads a pre-trained YOLOv8 model for object detection, and uses the Supervision library for annotating the detected objects on the image. The class also handles the transformation of the laser scan data based on the robot's pose.
        """
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
        # sub for depth image
        rospy.Subscriber(
            "/realsense/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=1,
        )

        self.camera_fx = 614.9791259765625
        self.camera_fy = 615.01416015625
        self.camera_cx = 430.603271484375
        self.camera_cy = 237.27053833007812

        self.robot_position = None
        self.robot_orientation = None
        self.depth_dict = dict()
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

        if msg.header.stamp in self.depth_dict:
            depth_image = self.depth_dict[msg.header.stamp]
            # it returns a generator but it only contains one element
            for boxes, mask, conf_score, _, _, class_name in detections[0] if results else []:
                if conf_score > 0.5:
                    # get the bounding box
                    x1, y1, x2, y2 = boxes
                    # get the center of the bounding box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    # get the depth value at the center of the bounding box
                    depth = depth_image[int(y_center), int(x_center)] / 1000
                    if depth > 0:
                        rospy.loginfo(f"Distance to object {class_name}: {depth} m")

    def depth_callback(self, msg):
        # depth image contains values ranging from 0.0 to 2999.0, those are the mm per pixel
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_image = np.array(depth_image, dtype=np.float32)
        # Idea: use a dict to keep track of images for synchronization:
        # images = {seqN / timestamp: image}
        # with depth you just access the data in O(1) time with the key
        # to be considered: 30FPS -> the dict should be cleared after a certain dimension
        self.depth_dict[msg.header.stamp] = depth_image

    def scan_callback(self, msg):
        pass


if __name__ == "__main__":
    node = Node()
