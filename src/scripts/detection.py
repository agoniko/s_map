#! /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python
import sys

# Add the directory containing your ROS package's generated message Python files to sys.path
sys.path.append("/Users/nicoloagostara/catkin_ws/src/s_map/msg")
# ros packages
import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from s_map.msg import Detection

# from s_map.msg import Detection, Box, Mask

# others
# import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from ctypes import *
from collections import deque

RGB_TOPIC = "/realsense/rgb/image_raw"
DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
SCAN_TOPIC = "/scan"
MODEL_PATH = rospkg.RosPack().get_path("s_map") + "/models/yolov8m-seg.pt"


class Node(object):

    __slots__ = [
        "cv_bridge",
        "device",
        "detector",
        "mask_annotator",
        "label_annotator",
        "box_annotator",
        "image_sub",
        "depth_sub",
        "laser_sub",
        "annotated_images",
        "annotated_image_pub",
        "results",
        "results_pub",
        "synchronizer",
        "pub_timer",
    ]
    cv_bridge: CvBridge
    device: torch.device
    detector: YOLO
    mask_annotator: sv.MaskAnnotator
    label_annotator: sv.LabelAnnotator
    box_annotator: sv.BoxCornerAnnotator
    detections_dict: dict
    image_sub: rospy.Subscriber
    annotated_images: deque
    annotated_image_pub: rospy.Publisher
    results: deque
    results_pub: rospy.Publisher
    pub_timer: rospy.Timer

    def __init__(self):
        """
        Initializes a ROS node for object segmentation and laser scan processing.
        It synchronize messages from the RGB camera, the depth camera and the LaserScan, performs detection and publish the results.
        Results are then processed by mapping module to create a 3D semantic map.
        """
        rospy.init_node("detection")
        self.cv_bridge = CvBridge()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
        )
        rospy.loginfo(f"Using {self.device} device for detection")

        # segmentation and detection model
        self.detector = YOLO(MODEL_PATH)
        self.detector = self.detector.to(self.device)

        # Using supervision library for easier annotation
        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxCornerAnnotator()

        # subscribers
        self.image_sub = rospy.Subscriber(
            RGB_TOPIC, Image, self.detection_callback, queue_size=10
        )

        # publishers
        # One to show the annotated image (for testing purposes)
        # One to publish the results of the detection (boxes, masks, labels, synch. depth image)
        self.annotated_image_pub = rospy.Publisher(
            "/s_map/detection/image_annotated", Image, queue_size=10
        )
        self.results_pub = rospy.Publisher(
            "/s_map/detection/results", Detection, queue_size=10
        )

        self.annotated_images = deque(maxlen=30)
        self.results = deque(maxlen=30)

        self.pub_timer = rospy.Timer(rospy.Duration(1.0 / 30.0), self.publish_results)

        rospy.spin()

    def publish_results(self, event):
        """
        Publish annotated images and detection results
        """
        start = time.time()
        if self.annotated_images:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(
                self.annotated_images.popleft(), "rgb8"
            )
            self.annotated_image_pub.publish(annotated_image_msg)

        if self.results:
            self.results_pub.publish(self.results.popleft())

        end = time.time()
        # print("Publish time: ", end - start)

    def preprocess_msg(self, detections, header):
        """
        Preprocess the detection message to be published from supervision detection
        """
        try:
            boxes = detections.xyxy.astype(np.int32)
            boxes = boxes.flatten()  # already rescaled to the original image size
            labels = detections.data["class_name"]
            track_id = detections.tracker_id.astype(np.int32)
            conf = detections.confidence

            masks = detections.mask
            masks = np.moveaxis(masks, 0, -1).astype(np.uint8)
            mask_msg = self.cv_bridge.cv2_to_imgmsg(masks, "passthrough")

            res = Detection()
            res.header = header
            res.boxes = boxes
            res.ids = track_id
            res.scores = conf
            res.masks = mask_msg
            res.labels = labels

            return res
        except:
            return None

    def detection_callback(self, image_msg):
        frame = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # performing object detection/segmentation with YOLO
        results = next(
            self.detector.track(
                frame,
                device=self.device,
                stream=True,
                conf=0.5,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
        )
        # remember that YOLO rescale output to 384, 640 while frame has a different dimension
        # for this reason we take the normalized coordinates
        # CameraPoseEstimator handle this issue
        if results:
            detections = sv.Detections.from_ultralytics(results)
            labels = detections.data["class_name"]
            track_ids = detections.tracker_id

            if track_ids is None or labels is None:
                return

            labels = [
                f"{track_id}: {label}" for track_id, label in zip(track_ids, labels)
            ]
            # annotating frame for testing purposes
            frame = self.mask_annotator.annotate(frame, detections)
            frame = self.label_annotator.annotate(frame, detections, labels)
            frame = self.box_annotator.annotate(frame, detections)
            self.annotated_images.append(frame)

            detection_msg = self.preprocess_msg(detections, image_msg.header)
            if detection_msg:
                self.results.append(detection_msg)


if __name__ == "__main__":
    node = Node()
