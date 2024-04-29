#!/usr/bin/env python3
import rospy
from rtabmap_ros.msg import RGBDImage
from sensor_msgs.msg import Image, CameraInfo


class Node:
    def __init__(self):
        self.frontleft = rospy.Subscriber("/frontleft/rgbd_image", RGBDImage, self.callback, queue_size=100)
        self.frontright = rospy.Subscriber("/frontright/rgbd_image", RGBDImage, self.callback, queue_size=100)
        self.back = rospy.Subscriber("/back/rgbd_image", RGBDImage, self.callback, queue_size=100)
        self.right = rospy.Subscriber("/right/rgbd_image", RGBDImage, self.callback, queue_size=100)
        self.left = rospy.Subscriber("/left/rgbd_image", RGBDImage, self.callback, queue_size=100)

        # create an rgb pub and a depth pub for each of the subscribers
        self.rgb_pub = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=100)
        self.depth_pub = rospy.Publisher("/camera/depth/image_raw", Image, queue_size=100)
        self.rgb_info_pub = rospy.Publisher("/camera/rgb/camera_info", CameraInfo, queue_size=100)
        self.depth_info_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=100)
        

        rospy.init_node("test_node")
        rospy.spin()
    
    def callback(self, msg):
        print("RGB:", msg.rgb.width, msg.rgb.height)
        print("Depth:", msg.depth.width, msg.depth.height)
        self.rgb_pub.publish(msg.rgb)
        self.depth_pub.publish(msg.depth)
        self.rgb_info_pub.publish(msg.rgb_camera_info)
        self.depth_info_pub.publish(msg.depth_camera_info)




if __name__ == "__main__":
    Node()
