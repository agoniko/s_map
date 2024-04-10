#! /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ros packages
import rospy
import rospkg
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped


# others
# import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from ctypes import *
from geometric_transformations import Transformer, CameraPoseEstimator, TransformHelper
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
            "/realsense/rgb/image_raw", Image, self.detection_callback, queue_size=10
        )
        self.pub = rospy.Publisher(
            "/s_map/detection/image_annotated", Image, queue_size=1
        )

        # pub_sub for scan
        rospy.Subscriber("/b_scan", LaserScan, self.scan_callback, queue_size=1)
        self.pub_scan = rospy.Publisher("/s_map/scan", LaserScan, queue_size=1)
        self.transformer = TransformHelper()  # 30 secs cache time default

        # sub for robot orientation
        self.pose_sub = rospy.Subscriber(
            "/robot_pose", Pose, self.pose_callback, queue_size=1
        )
        # sub for depth image
        rospy.Subscriber(
            "/realsense/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=10,
        )

        self.camera_fx = 614.9791259765625
        self.camera_fy = 615.01416015625
        self.camera_cx = 430.603271484375
        self.camera_cy = 237.27053833007812

        self.robot_position = None
        self.robot_orientation = None
        self.depth_dict = dict()
        self.detecions_dict = dict()

        self.pose_estimator = CameraPoseEstimator(
            "/realsense/aligned_depth_to_color/camera_info"
        )
        self.tf_transformer = Transformer()
        rospy.Timer(rospy.Duration(5), self.publish_objects)
        #rospy.Timer(rospy.Duration(1), self.test_tf)
        rospy.spin()

    def test_tf(self, event):
        # Example timestamp (stamp) of the generated point
        stamp = list(self.depth_dict.keys())[-1]  # or any specific timestamp
        point_source = PointStamped()
        point_source.header.frame_id = "realsense_rgb_optical_frame"
        point_source.point.x = 1.0
        point_source.point.y = 2.0
        point_source.point.z = 0.0

        # Lookup transform and transform coordinates using the timestamp
        transformed_point = self.transformer.lookup_transform_and_transform_coordinates(
            "realsense_rgb_optical_frame", "map", point_source, stamp
        )

        if transformed_point:
            rospy.loginfo(
                "Transformed coordinates: (%f, %f, %f) in frame: %s",
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z,
                "map",
            )
        else:
            rospy.logerr("Failed to transform coordinates.")

    def pose_callback(self, msg):
        self.robot_orientation = msg.orientation
        self.robot_position = msg.position

    def detection_callback(self, msg):
        print("Detection callback: ", msg.header.frame_id)
        start = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = next(
            self.detector(
                frame, device=self.device, stream=True, conf=0.4, verbose=False
            )
        )

        # remember that YOLO rescale output to 384, 640 while frame has a different dimension
        detections = sv.Detections.from_ultralytics(results)
        boxes = results.boxes.xyxyn
        masks = detections.mask
        conf_scores = detections.confidence
        labels = detections.data["class_name"]

        self.detecions_dict[msg.header.stamp] = (boxes, labels)

    def depth_callback(self, msg):
        #print frame id
        #print(msg.header.frame_id)
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

    def publish_objects(self, event):
        # boxes: np.array (n, 4) normalized between 0 and 1
        pub = rospy.Publisher("/s_map/objects", Marker, queue_size=1)
        marker_msg = Marker()
        marker_msg.header.frame_id = "world"  # Specify the frame in which the point is defined
        marker_msg.ns = "my_namespace"
        marker_msg.type = Marker.POINTS
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.5  # Size of the points
        marker_msg.scale.y = 0.5
        marker_msg.scale.z = 0.5
        marker_msg.color.a = 1.0  # Alpha (transparency)

        # iterating over list allow pop
        for stamp in list(self.detecions_dict.keys()):
            if stamp not in self.depth_dict:
                continue

            # removing objects from the dict
            boxes, labels = self.detecions_dict.pop(stamp)
            depth_image = self.depth_dict.pop(stamp)

            marker_msg.header.stamp = stamp
            i = 0
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                # get the center of the bounding box
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                # get the depth value at the center of the bounding box
                depth = depth_image[int(y_center), int(x_center)] / 1000
                marker_msg.id = stamp.secs + i # An id for the marker
                i += 1
                if depth > 0:
                    x_3d, y_3d = self.pose_estimator.normalized_pixel_to_3d(
                        x_center, y_center, depth
                    )
                    point_source = PointStamped()
                    point_source.header.frame_id = "realsense_rgb_optical_frame"
                    point_source.point.x = x_3d
                    point_source.point.y = y_3d
                    point_source.point.z = depth
                    transformed_point = (
                        self.transformer.lookup_transform_and_transform_coordinates(
                            "realsense_rgb_optical_frame", "world", point_source, stamp
                        )
                    )
                    print(transformed_point.point)
                    point = transformed_point.point

                    marker_msg.color.r = 0.0
                    marker_msg.color.g = 0.0
                    marker_msg.color.b = 0.0

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

                    marker_msg.points.append(point)
                    pub.publish(marker_msg)
                    marker_msg.points = []
                    print("Published objects")


if __name__ == "__main__":
    node = Node()
