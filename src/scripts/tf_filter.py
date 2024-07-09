#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
from nav_msgs.msg import Odometry

import numpy as np

class TfFilterNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Define the list of frame ids to filter out
        self.unwanted_frames = [""]
        self.rename_frames = []

        # Subscribe to the /tf topic
        rospy.Subscriber('/tf_old', tf2_msgs.msg.TFMessage, self.tf_callback)
        rospy.Subscriber('/tf_static_old', tf2_msgs.msg.TFMessage, self.tf_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        self.odom_pub = rospy.Publisher("/odom/filtered", Odometry, queue_size=10)
    
    def tf_callback(self, msg):
        filtered_transforms = []
        
        for transform in msg.transforms:
            if transform.child_frame_id not in self.unwanted_frames and transform.header.frame_id not in self.unwanted_frames:
                filtered_transforms.append(transform)
            
            if transform.header.frame_id in self.rename_frames:
                transform.header.frame_id = transform.header.frame_id + "_new"
                filtered_transforms.append(transform)
        
        if filtered_transforms:
            filtered_msg = tf2_msgs.msg.TFMessage(transforms=filtered_transforms)
            self.tf_broadcaster.sendTransform(filtered_msg.transforms)
    
    def odom_callback(self, msg):
        #overwrite the odom covariance matrix as 0.1 Identity
        #msg.pose.covariance = (np.identity(6)*0.1).flatten().tolist()
        #msg.twist.covariance = (np.identity(6)*0.1).flatten().tolist()
        self.odom_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('tf_filter')
    node = TfFilterNode()
    rospy.spin()
