#! /usr/bin/env python3

import rospy
from tf2_msgs.msg import TFMessage

class TransformListener:
    def __init__(self):
        self.transforms = {}
        rospy.init_node('tf_static_listener', anonymous=True)
        rospy.Subscriber('/tf_static', TFMessage, self.tf_static_callback)
        rospy.spin()

    def tf_static_callback(self, msg):
        for transform in msg.transforms:
            #if "mir" in transform.header.frame_id or "ur5e" in transform.header.frame_id or "realsense" in transform.header.frame_id:
            if "realsense" in transform.header.frame_id:
                key = (transform.header.frame_id, transform.child_frame_id)
                if key not in self.transforms:
                    self.transforms[key] = transform
                    #write a message as x y z roll pitch yaw parent child
                    print(f"{transform.transform.translation.x} {transform.transform.translation.y} {transform.transform.translation.z} {transform.transform.rotation.x} {transform.transform.rotation.y} {transform.transform.rotation.z} {transform.transform.rotation.w} {transform.header.frame_id} {transform.child_frame_id}")

if __name__ == '__main__':
    TransformListener()
