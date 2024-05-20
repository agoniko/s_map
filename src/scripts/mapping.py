#! /usr/bin/env python3
# /Users/nicoloagostara/miniforge3/envs/ros_env/bin/python

# ROS packages
import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray
from s_map.msg import Detection
import rospkg

# Local modules
from utils import (
    create_marker_array,
    create_delete_marker,
    create_pointcloud_message,
    time_it,
)
import supervision as sv
import numpy as np
from geometric_transformations import CameraPoseEstimator, TransformHelper
from world import World, Obj
from pose_reliability import ReliabilityEvaluator
import open3d as o3d


# Topic constants
RESULT_TOPIC = "/s_map/detection/results"
CAMERA_INFO_TOPIC = "/aligned_depth/camera_info"
SCAN_TOPIC = "/scan"
MARKERS_TOPIC = "/s_map/objects"
PC_TOPIC = "/s_map/pointcloud"

# Frame constants
WORLD_FRAME = "vision"
PKG_PATH = rospkg.RosPack().get_path("s_map")

class Mapper(object):
    def __init__(self):
        rospy.init_node("mapping", anonymous=True)
        self.cv_bridge = CvBridge()
        self.init_subscribers()
        self.init_publishers()
        self.pose_estimator_dict = {}
        self.transformer = TransformHelper()
        self.world = World()
        self.pose_reliability_evaluator = {}
        rospy.loginfo("Mapping node initialized")
        rospy.Timer(rospy.Duration(0.2), self.check_still_there)

    def init_subscribers(self):
        self.result_subscriber = rospy.Subscriber(
            RESULT_TOPIC, Detection, self.process_data, queue_size=50
            )

    def init_publishers(self):
        self.marker_pub = rospy.Publisher(MARKERS_TOPIC, MarkerArray, queue_size=10)
        self.pc_pub = rospy.Publisher(PC_TOPIC, PointCloud2, queue_size=2)

    # @time_it
    def preprocess_msg(self, msg: Detection):
        header = msg.header
        labels = np.array(msg.labels)
        n = len(labels)
        boxes = np.array(msg.boxes).reshape(n, 4)
        scores = np.array(msg.scores)
        ids = np.array(msg.ids)
        depth_image = self.cv_bridge.imgmsg_to_cv2(msg.depth, "passthrough")

        masks = self.cv_bridge.imgmsg_to_cv2(msg.masks)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=2)
        masks = np.moveaxis(masks, 2, 0)
        masks = masks.astype(np.uint8)

        if masks.shape[0] != n:
            masks = [None] * n
        
        return header, boxes, labels, scores, ids, masks, depth_image, msg.camera_name
    
    def is_reliable(self, header):
        if header.frame_id not in self.pose_reliability_evaluator:
            self.pose_reliability_evaluator[header.frame_id] = ReliabilityEvaluator(header.frame_id, WORLD_FRAME)
        return self.pose_reliability_evaluator[header.frame_id].evaluate(header.stamp)
    
    def get_pose_estimator(self, camera_name):
        if camera_name not in self.pose_estimator_dict.keys():
            self.pose_estimator_dict[camera_name] = CameraPoseEstimator(camera_name + CAMERA_INFO_TOPIC)
        return self.pose_estimator_dict[camera_name]

    # @time_it
    def process_data(self, detection):
        """
        Process the data received from detection and depth sensors.

        Args:
            detection: The detection message containing object information.
            depth: The depth image message.

        Returns:
            None
        """
        if not self.is_reliable(detection.header):
            return
        
        header, boxes, labels, scores, ids, masks, depth_image, camera_name = self.preprocess_msg(detection)
        pose_estimator = self.get_pose_estimator(camera_name)

        for id, box, label, mask, score in zip(ids, boxes, labels, masks, scores):
            obj = self.compute_object(
                id, box, depth_image, mask, label, score, header, pose_estimator
            )
            if obj is None:
                continue

            self.world.manage_object(obj)
        self.publish_markers(header.stamp)
        self.publish_pointclouds(WORLD_FRAME, header.stamp)

    #@time_it
    def check_still_there(self, event):
        """
        This function checks if the objects saved in the world, that now should be infront of the camera, are still there.
        """
        for camera_frame in self.pose_reliability_evaluator.keys():
            point = np.array([[0, 0, 1.5]]) # 1.5 meters in front of the camera
            point_world_frame = self.transformer.fast_transform(camera_frame, WORLD_FRAME, point, rospy.Time.now())
            if point_world_frame is None:
                return
            objects = self.world.query_by_distance(point_world_frame[0], 1.5)
            to_remove = []  
            for obj in objects:
                if obj.last_seen.to_sec() < rospy.Time.now().to_sec() - 5.0:
                    to_remove.append(obj.id)
            
            self.world.remove_objects(to_remove)

    # @time_it
    def compute_object(self, id, box, depth_image, mask, label, score, header, pose_estimator):
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
        pc = self.compute_pointcloud(depth_image, mask)
        if len(pc) < 200:
            return None

        pc_camera_frame = pose_estimator.multiple_pixels_to_3d(pc)
        pc_world_frame = self.transformer.fast_transform(
            header.frame_id, WORLD_FRAME, pc_camera_frame, header.stamp
        )
        obj = Obj(id, pc_world_frame, label, score, header.stamp)
        return obj

    # @time_it
    def compute_pointcloud(self, depth_image, mask):
        pc = depth_image * mask
        (ys, xs) = np.argwhere(pc).T

        #remove from the pointcloud the points that are too close or too far from the camera
        zs = pc[ys, xs] / 1000
        mask = np.logical_and(zs > 0.5, zs < 3.0)
        ys = ys[mask]
        xs = xs[mask]
        zs = zs[mask]

        pointcloud = np.array([xs, ys, zs]).T
        return pointcloud

    # @time_it
    def publish_pointclouds(self, frame, stamp):
        objects = self.world.get_objects()
        msg = create_pointcloud_message(objects, frame, stamp)
        self.pc_pub.publish(msg)

    # @time_it
    def publish_markers(self, stamp):
        marker = create_delete_marker(WORLD_FRAME)
        self.marker_pub.publish(marker)
        objects = self.world.get_objects()
        # for obj in objects:
        #    points = np.asarray(obj.pcd.points)
        #    np.savetxt(PKG_PATH + f"/pc/{obj.id}_{obj.label}.txt", points, delimiter=",")
        msg = create_marker_array(objects, WORLD_FRAME, stamp)
        if msg:
            self.marker_pub.publish(msg)


if __name__ == "__main__":
    mapper = Mapper()
    rospy.spin()
