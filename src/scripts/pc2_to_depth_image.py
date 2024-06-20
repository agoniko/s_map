#! /usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from geometric_transformations import CameraPoseEstimator, TransformHelper
from utils import time_it



class PointCloudToDepthImage:
    def __init__(self):
        self.initialize_params()
        self.bridge = CvBridge()
        self.camera_pose_estimator = CameraPoseEstimator(self.camera_info_topic)
        self.transform_helper = TransformHelper()
        self.pointcloud_sub = rospy.Subscriber('/rtabmap/cloud_map', PointCloud2, self.pointcloud_callback)
        self.image_sub = rospy.Subscriber(self.rgb_topic, Image, self.image_callback)
        self.depth_image_pub = rospy.Publisher('/processed_depth_image', Image, queue_size=1)

        self.points = None
        self.pc_frame = None
    
    def initialize_params(self):
        self.camera_info_topic = rospy.get_param("~camera_info_topic", None)
        self.rgb_topic = rospy.get_param("~rgb_topic", None)
        if not self.camera_info_topic:
            rospy.logerr("Camera info topic not specified")
            rospy.signal_shutdown("Camera info topic not specified")
        
        if not self.rgb_topic:
            rospy.logerr("RGB topic not specified")
            rospy.signal_shutdown("RGB topic not specified")
    
    def pointcloud_callback(self, msg):
        pc = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc))
        if points.shape[0] == 0:
            return
        
        self.points = points
        self.pc_frame = msg.header.frame_id
    
    @time_it
    def image_callback(self, msg):
        camera_frame = msg.header.frame_id
        stamp = msg.header.stamp
        if self.points is None:
            return
        if self.points is None or self.points.shape[0] == 0:
            return
        camera_frame_points = self.transform_helper.fast_transform(self.pc_frame, camera_frame, self.points, stamp)
        if camera_frame_points is None:
            return
        
        depth_image = self.camera_pose_estimator.multiple_3d_to_depth_image(camera_frame_points)
        
        depth_image = self.camera_pose_estimator.multiple_3d_to_depth_image(camera_frame_points)
        if depth_image is None:
            return

        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding = "passthrough")
        depth_image_msg.header = msg.header
        self.depth_image_pub.publish(depth_image_msg)


    

if __name__ == '__main__':
    rospy.init_node('pointcloud_to_depth_image')
    converter = PointCloudToDepthImage()
    rospy.spin()
