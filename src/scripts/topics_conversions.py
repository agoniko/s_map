#!/usr/bin/env python3
import rospy
from rtabmap_ros.msg import RGBDImage
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import numpy as np

DEPTH_TOPIC = "/realsense/aligned_depth_to_color/image_raw"
RGB_TOPIC = "/realsense/rgb/image_raw"
CAMERA_INFO_TOPIC = "/realsense/aligned_depth_to_color/camera_info"

class Node:
    def __init__(self):
        #self.frontleft = rospy.Subscriber("/frontleft/rgbd_image", RGBDImage, self.callback, queue_size=100)
        #self.frontright = rospy.Subscriber("/frontright/rgbd_image", RGBDImage, self.callback, queue_size=100)
        self.back = Subscriber("/spot/camera/back/image", Image)
        self.back_depth = Subscriber("/spot/depth/back/image", Image)
        self.back_info = Subscriber("/spot/camera/back/camera_info", CameraInfo)

        self.syncronizer = TimeSynchronizer(
            [self.back, self.back_depth, self.back_info], 1000
        )
        self.syncronizer.registerCallback(self.callback)
        self.cv_bridge = CvBridge()


        #self.right = rospy.Subscriber("/right/rgbd_image", RGBDImage, self.callback, queue_size=100)
        #elf.left = rospy.Subscriber("/left/rgbd_image", RGBDImage, self.callback, queue_size=100)

        # create an rgb pub and a depth pub for each of the subscribers
        #self.rgb_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=100)
        #self.depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=100)
        #self.rgb_info_pub = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, queue_size=100)
        #self.depth_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=100)

        self.rgb_pub = rospy.Publisher(RGB_TOPIC, Image, queue_size=100)
        self.depth_pub = rospy.Publisher(DEPTH_TOPIC, Image, queue_size=100)
        self.rgb_info_pub = rospy.Publisher(CAMERA_INFO_TOPIC, CameraInfo, queue_size=100)
        
        rospy.init_node("test_node")
        rospy.spin()
    
    def callback(self, rgb, depth, info):

        d = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")
        print(d.shape)
        print(np.min(d), np.max(d))

        return
        self.rgb_pub.publish(rgb)
        self.depth_pub.publish(depth)
        self.rgb_info_pub.publish(info)
        rospy.loginfo("Published")


if __name__ == "__main__":
    Node()
