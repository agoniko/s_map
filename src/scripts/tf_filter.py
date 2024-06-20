#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg

class TfFilterNode:
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Define the list of frame ids to filter out
        self.unwanted_frames = ['map']

        # Subscribe to the /tf topic
        rospy.Subscriber('/tf_old', tf2_msgs.msg.TFMessage, self.tf_callback)
        rospy.Subscriber('/tf_static_old', tf2_msgs.msg.TFMessage, self.tf_callback)
    
    def tf_callback(self, msg):
        filtered_transforms = []
        
        for transform in msg.transforms:
            if transform.header.frame_id.lstrip("/") == "mir/base_footprint":
                print(transform)
            if transform.child_frame_id not in self.unwanted_frames and transform.header.frame_id not in self.unwanted_frames:
                filtered_transforms.append(transform)
        
        if filtered_transforms:
            filtered_msg = tf2_msgs.msg.TFMessage(transforms=filtered_transforms)
            self.tf_broadcaster.sendTransform(filtered_msg.transforms)

if __name__ == '__main__':
    rospy.init_node('tf_filter')
    node = TfFilterNode()
    rospy.spin()
