#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class DepthImageClipper:
    def __init__(self):
        self.bridge = CvBridge()
        self.raw_depth_topic = rospy.get_param("~raw_depth_topic", None)
        self.cleaned_depth_topic = rospy.get_param("~cleaned_depth_topic", None)
        if self.raw_depth_topic is None or self.cleaned_depth_topic is None:
            rospy.logerr("Raw depth topic or cleaned depth topic not provided")
            rospy.signal_shutdown("Raw depth topic or cleaned depth topic not provided")

        self.subscriber = rospy.Subscriber(self.raw_depth_topic, Image, self.callback)
        self.publisher = rospy.Publisher(self.cleaned_depth_topic, Image, queue_size=10)
        self.max_depth = 9000  # Maximum depth in meters

    def callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

            # Clamp depth values at 5 meters
            clipped_image = np.clip(cv_image, 0, self.max_depth)
            clipped_image = cv2.erode(clipped_image, np.ones((3, 3)), iterations=3).astype(np.uint16)

            # Convert OpenCV image back to ROS Image message
            clipped_image_msg = self.bridge.cv2_to_imgmsg(clipped_image, encoding="passthrough")
            clipped_image_msg.header = data.header

            # Publish the clipped depth image
            self.publisher.publish(clipped_image_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('depth_image_clipper', anonymous=True)
    DepthImageClipper()
    rospy.spin()