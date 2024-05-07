#! /usr/bin/env python3
# /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ROS packages
import rospy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber
from visualization_msgs.msg import MarkerArray
from s_map.msg import Detection
import rospkg

# Local modules
from utils import (
    create_marker_array,
    get_vercitces,
    bbox_iou,
    create_delete_marker,
    delete_marker,
)
import supervision as sv
import numpy as np
from geometric_transformations import CameraPoseEstimator, TransformHelper
from world import World, Obj
from pose_reliability import ReliabilityEvaluator
import open3d as o3d



# Topic constants
RESULT_TOPIC = "/s_map/detection/results"
DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
RGB_TOPIC = "/realsense/rgb/image_raw"
CAMERA_INFO_TOPIC = "/realsense/aligned_depth_to_color/camera_info"
SCAN_TOPIC = "/scan"
MARKERS_TOPIC = "/s_map/objects"

# Frame constants
WORLD_FRAME = "world"
CAMERA_FRAME = "realsense_rgb_optical_frame"


class Mapper(object):
    def __init__(self):
        rospy.init_node("mapping", anonymous=True)
        self.cv_bridge = CvBridge()
        self.init_subscribers()
        self.init_publishers()
        self.pose_estimator = CameraPoseEstimator(CAMERA_INFO_TOPIC)
        self.transformer = TransformHelper()
        self.world = World()
        self.pose_reliability_evaluator = ReliabilityEvaluator(
            CAMERA_FRAME, WORLD_FRAME
        )

    def init_subscribers(self):
        self.result_subscriber = Subscriber(RESULT_TOPIC, Detection)
        self.depth_subscriber = Subscriber(DEPTH_TOPIC, Image)
        self.laser_subscriber = Subscriber(SCAN_TOPIC, LaserScan)
        self.synchronizer = TimeSynchronizer(
            [self.result_subscriber, self.depth_subscriber], 1000
        )
        self.synchronizer.registerCallback(self.process_data)

    def init_publishers(self):
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, MarkerArray, queue_size=10)

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

    def process_data(self, detection, depth):
        """
        Process the data received from detection and depth sensors.

        Args:
            detection: The detection message containing object information.
            depth: The depth image message.

        Returns:
            None
        """
        start = rospy.get_time()
        if not self.pose_reliability_evaluator.evaluate(detection.header.stamp):
            # rospy.logwarn("Pose is not reliable")
            return
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")
        header, boxes, labels, scores, ids, masks = self.preprocess_msg(detection)

        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            obj = self.compute_object(
                id, box, depth_image, mask, label, score, header.stamp
            )
            if obj is None:
                continue
            self.world.manage_object(obj)
        self.publish_markers(header.stamp)
        # print("Time to process detection: ", rospy.get_time() - start)

    def compute_object(self, id, box, depth_image, mask, label, score, stamp):
        """
        Computes the object information in the world frame
        Args:
            id (int): The ID of the object.
            box (tuple): The bounding box coordinates (xmin, ymin, xmax, ymax).
            depth_image (numpy.ndarray): The depth image.
            mask (numpy.ndarray): The mask indicating the object region.
            label (str): The label of the object.
            score (float): The score of the object.
            stamp (float): The timestamp of the object.

        Returns:
            Obj: An instance of the Obj class representing the computed object.
                Returns None if the computation fails.
        """
        xmin, ymin, xmax, ymax = box
        filtered_depth = depth_image * mask
        filtered_depth = filtered_depth[filtered_depth != 0]

        if len(filtered_depth) == 0 or xmin < 0 or ymin < 0 or xmax > 848 or ymax > 480:
            return None
        
        pc = self.compute_pointcloud(depth_image, mask)
        pc_camera_frame = self.pose_estimator.multiple_pixels_to_3d(pc)
        bbox_camera_frame = self.extract_bbox_vertices_from_pointcloud(pc_camera_frame)

        bbox_world_frame = self.transformer.transform_coordinates(
            CAMERA_FRAME, WORLD_FRAME, bbox_camera_frame, stamp
        )
        return Obj(id, bbox_world_frame, label, score)
    
    def compute_pointcloud(self, depth_image, mask):
        pc = depth_image * mask
        (ys, xs) = np.argwhere(pc).T
        zs = pc[ys, xs] / 1000 # convert to meters
        pointcloud = np.array([xs, ys, zs]).T
        return pointcloud


    def extract_bbox_vertices_from_pointcloud(self, pointcloud):
        path = rospkg.RosPack().get_path("s_map") + "/pointcloud.txt"
        np.savetxt(path, pointcloud, delimiter=",")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd, _ = pcd.remove_radius_outlier(nb_points=15, radius=0.05)
        bbox = pcd.get_axis_aligned_bounding_box()
        vertices = np.asarray(bbox.get_box_points())

        return vertices


    def publish_markers(self, stamp):
        marker = create_delete_marker(WORLD_FRAME)
        self.marker_pub.publish(marker)
        objects = self.world.get_objects()
        msg = create_marker_array(objects, WORLD_FRAME, stamp)
        if msg:
            self.marker_pub.publish(msg)


if __name__ == "__main__":
    mapper = Mapper()
    rospy.spin()
