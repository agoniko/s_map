#! /usr/bin/env python3
# /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from s_map.msg import Detection
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
from geometric_transformations import CameraPoseEstimator

# Global Configuration Variables
RGB_TOPIC = "/realsense/rgb/image_raw"
DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
SCAN_TOPIC = "/scan"
DETECTION_RESULTS_TOPIC = "/s_map/detection/results"
ANNOTATED_IMAGES_TOPIC = "/s_map/annotated_images"
MODEL_PATH = rospkg.RosPack().get_path("s_map") + "/models/yolov8m-seg.pt"
DETECTION_CONFIDENCE = 0.5
TRACKER = "bytetrack.yaml"
SUBSCRIPTION_QUEUE_SIZE = 50
CAMERA_INFO_TOPIC = "/realsense/aligned_depth_to_color/camera_info"



class Node:
    """
    A ROS node for object detection using YOLO, processing images and publishing results.

    Attributes:
        cv_bridge (CvBridge): Bridge between ROS Image messages and OpenCV formats.
        device (torch.device): Device configuration for PyTorch (either CUDA or CPU).
        detector (YOLO): Initialized YOLO model for object detection.
        mask_annotator (sv.MaskAnnotator): Annotator for applying masks on detected objects.
        label_annotator (sv.LabelAnnotator): Annotator for labeling detected objects.
        box_annotator (sv.BoxCornerAnnotator): Annotator for drawing bounding boxes around detected objects.
        image_sub (rospy.Subscriber): Subscriber to the RGB image topic.
        depth_sub (rospy.Subscriber): Subscriber to the depth image topic.
        laser_sub (rospy.Subscriber): Subscriber to the laser scan topic.
        annotated_image_pub (rospy.Publisher): Publisher for annotated images.
        results_pub (rospy.Publisher): Publisher for detection results.
    """

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
        "annotated_image_pub",
        "results_pub",
    ]

    def __init__(self):
        """Initialize the node, its publications, subscriptions, and model."""
        rospy.init_node("detection_node")
        self.cv_bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.detector = YOLO(MODEL_PATH)
        self.detector = self.detector.to(self.device)
        rospy.loginfo(f"Using device: {self.device}")

        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxCornerAnnotator()

        # Subscribers
        self.image_sub = rospy.Subscriber(
            RGB_TOPIC,
            Image,
            self.detection_callback,
            queue_size=SUBSCRIPTION_QUEUE_SIZE,
        )
        # self.depth_sub = rospy.Subscriber(DEPTH_TOPIC, Image, self.depth_callback, queue_size=SUBSCRIPTION_QUEUE_SIZE)
        # self.laser_sub = rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback, queue_size=SUBSCRIPTION_QUEUE_SIZE)

        # Publishers
        self.annotated_image_pub = rospy.Publisher(
            ANNOTATED_IMAGES_TOPIC, Image, queue_size=10
        )
        self.results_pub = rospy.Publisher(
            DETECTION_RESULTS_TOPIC, Detection, queue_size=10
        )

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
        """
        Callback for processing images received from the RGB topic.
        Received images are already rectified and aligned with the depth image.
        Performs detection, annotations, and publishes the results.

        Args:
            image_msg (Image): The incoming ROS message containing the image data.
        """
        frame = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        results = next(
            self.detector.track(
                frame,
                device=self.device,
                conf=DETECTION_CONFIDENCE,
                stream=True,
                persist=True,
                tracker=TRACKER,
                verbose=False,
            )
        )
        if results:
            detections = sv.Detections.from_ultralytics(results)
            frame = self.annotate_frame(frame, detections, image_msg.header)
            self.publish_results(detections, image_msg.header, frame)

    def annotate_frame(self, frame, detections, header):
        """
        Annotates the frame with masks, labels, and boxes based on detections.

        Args:
            frame (np.array): Image frame to annotate.
            detections (sv.Detections): Detection results.
            header (std_msgs.msg.Header): Header from the incoming image message.

        Returns:
            Image: Annotated image ready for publishing.
        """
        if (
            not detections
            or detections.data["class_name"] is None
            or detections.tracker_id is None
        ):
            return self.cv_bridge.cv2_to_imgmsg(frame, "rgb8", header)

        labels = [
            f"{tracker_id}:{class_name}"
            for class_name, tracker_id in zip(
                detections.data["class_name"], detections.tracker_id
            )
        ]

        frame = self.mask_annotator.annotate(frame, detections)
        frame = self.label_annotator.annotate(frame, detections, labels)
        frame = self.box_annotator.annotate(frame, detections)
        return self.cv_bridge.cv2_to_imgmsg(frame, "rgb8", header)

    def publish_results(self, detections, header, frame):
        """
        Publishes detection results and annotated images.

        Args:
            detections (sv.Detections): Detection results to be published.
            header (std_msgs.msg.Header): Header from the incoming image message, used for time stamping the published messages.
            frame (std_msgs.msg.Image): Annotated image to be published
        """
        detection_msg = self.preprocess_msg(detections, header)
        if detection_msg:
            self.results_pub.publish(detection_msg)
        self.annotated_image_pub.publish(frame)

    def depth_callback(self, msg):
        """Placeholder for processing depth images."""
        pass

    def scan_callback(self, msg):
        """Placeholder for processing laser scan data."""
        pass


if __name__ == "__main__":
    node = Node()
    rospy.spin()
