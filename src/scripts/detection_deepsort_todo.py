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
from message_filters import TimeSynchronizer, Subscriber
import cv2
import hashlib
from utils import time_it
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort



#Global Configuration Variables
DETECTION_RESULTS_TOPIC = "/s_map/detection/results"
ANNOTATED_IMAGES_TOPIC = "/s_map/annotated_images"
MODEL_PATH = rospkg.RosPack().get_path("s_map") + "/models/yolov8l-seg.pt"
DETECTION_CONFIDENCE = 0.6
TRACKER = "bytetrack.yaml"
SUBSCRIPTION_QUEUE_SIZE = 50

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
        "synchronizer",
        "annotated_image_pub",
        "results_pub",
        "camera_name",
        "tracker"
    ]

    def __init__(self):
        """Initialize the node, its publications, subscriptions, and model."""
        global RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, ANNOTATED_IMAGES_TOPIC
        rospy.init_node("detection_node")
        self.initialize_topics()

        self.cv_bridge = CvBridge()
        self.device = self.get_device()
        self.detector = YOLO(MODEL_PATH)
        self.detector = self.detector.to(self.device)
        self.tracker = DeepSort(max_age=30000, embedder="mobilenet", polygon=True)

        rospy.loginfo(f"Using device: {self.device}")

        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxCornerAnnotator()

        # Subscribers
        self.image_sub = Subscriber(RGB_TOPIC, Image)
        self.depth_sub = Subscriber(DEPTH_TOPIC, Image)
        self.synchronizer = TimeSynchronizer([self.image_sub, self.depth_sub], SUBSCRIPTION_QUEUE_SIZE)
        self.synchronizer.registerCallback(self.detection_callback)

        # Publishers
        self.annotated_image_pub = rospy.Publisher(
            ANNOTATED_IMAGES_TOPIC, Image, queue_size=10
        )
        self.results_pub = rospy.Publisher(
            DETECTION_RESULTS_TOPIC, Detection, queue_size=10
        )
    
    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def initialize_topics(self):
        global RGB_TOPIC, DEPTH_TOPIC, CAMERA_INFO_TOPIC, ANNOTATED_IMAGES_TOPIC
        RGB_TOPIC = rospy.get_param("~rgb_topic", None)
        DEPTH_TOPIC = rospy.get_param("~depth_topic", None)
        CAMERA_INFO_TOPIC = rospy.get_param("~camera_info_topic", None)
        ANNOTATED_IMAGES_TOPIC = rospy.get_param("~annotated_images_topic", "/s_map/annotated_images")
        self.camera_name = rospy.get_param("~camera_name", None)

        if not RGB_TOPIC or not DEPTH_TOPIC or not CAMERA_INFO_TOPIC or not self.camera_name:
            rospy.logerr("Missing required parameters. Exiting...")
            rospy.signal_shutdown("Missing required parameters. Exiting...")

    def preprocess_msg(self, detections, track_ids, depth, header):
        """
        Preprocess the detection message to be published from supervision detection
        """
        try:
            boxes = detections.xyxy.astype(np.int32)
            boxes = boxes.flatten()  # already rescaled to the original image size
            labels = detections.data["class_name"]
            #track_id = detections.tracker_id.astype(np.int32)
            conf = detections.confidence

            masks = detections.mask
            masks = np.moveaxis(masks, 0, -1).astype(np.uint8)

            mask_msg = self.cv_bridge.cv2_to_imgmsg(masks, "passthrough")
            res = Detection()
            res.header = header
            res.boxes = boxes
            res.ids = track_ids
            res.scores = conf
            res.masks = mask_msg
            res.labels = labels
            res.depth = depth
            res.camera_name = self.camera_name
            #rospy.loginfo(f"Publishing detection results for {self.camera_name} camera: ")
            #rospy.loginfo(f"{[(label, id) for label, id in zip(labels, res.ids)]}")

            return res
        except:
            return None
    
    def convert_detections(self, results):
        # Get the bounding boxes, labels and scores from the detections dictionary.
        class_dict = results.names
        cls = results.boxes.cls.cpu().numpy().astype(np.int32)
        polygons = [poly.reshape(-1) for poly in results.masks.xy]
        
        scores = results.boxes.conf.cpu().numpy()
        labels = np.array([class_dict[i] for i in cls])

        assert len(polygons) == len(labels) == len(scores)
        
        final_results = [polygons, labels, scores]

        return final_results
    
    def track(self, results, frame):
        detections = self.convert_detections(results)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        ids = []
        h, w = frame.shape[:2]
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            track_class = track.det_class
            x1, y1, x2, y2 = track.to_ltrb()
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            print(f"Track ID: {track_id}, Class: {track_class}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")
            ids.append(track_id)
        
        return ids
    
    #@time_it
    def detection_callback(self, image_msg, depth_msg):
        """
        Callback for processing images received from the RGB topic.
        Received images are already rectified and aligned with the depth image.
        Performs detection, annotations, and publishes the results.

        Args:
            image_msg (Image): The incoming ROS message containing the image data.
            depth_msg (Image): The incoming ROS message containing the depth data.
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
            track_ids = self.track(results, frame)
            print(results.boxes)
            if len(track_ids) != len(results.boxes):
                rospy.logerr("Number of track ids and detections do not match")
                return
            detections = sv.Detections.from_ultralytics(results)
            frame = self.annotate_frame(frame, detections, track_ids, image_msg.header)
            self.publish_results(detections, track_ids, depth_msg, depth_msg.header, frame)
        else:
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame, "rgb8", image_msg.header)
            self.annotated_image_pub.publish(img_msg)

    def annotate_frame(self, frame, detections, track_ids, header):
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
        ):
            return self.cv_bridge.cv2_to_imgmsg(frame, "rgb8", header)

        labels = [
            f"{tracker_id}:{class_name}"
            for class_name, tracker_id in zip(
                detections.data["class_name"], track_ids
            )
        ]

        frame = self.mask_annotator.annotate(frame, detections)
        frame = self.label_annotator.annotate(frame, detections, labels)
        frame = self.box_annotator.annotate(frame, detections)
        return self.cv_bridge.cv2_to_imgmsg(frame, "rgb8", header)

    def publish_results(self, detections, track_ids, depth, header, frame):
        """
        Publishes detection results and annotated images.

        Args:
            detections (sv.Detections): Detection results to be published.
            header (std_msgs.msg.Header): Header from the incoming image message, used for time stamping the published messages.
            depth (std_msgs.msg.Image): Depth image to be published associated to the detections
            frame (std_msgs.msg.Image): Annotated image to be published
        """
        detection_msg = self.preprocess_msg(detections, track_ids, depth, header)
        if detection_msg:
            self.results_pub.publish(detection_msg)
        self.annotated_image_pub.publish(frame)

if __name__ == "__main__":
    node = Node()
    rospy.spin()
