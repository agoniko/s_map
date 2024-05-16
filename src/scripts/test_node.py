#! /usr/bin/env python3

import message_filters
from sensor_msgs.msg import Image, CameraInfo
import rospy

def callback(image_rgb, camera_info_rgb, image_depth, camera_info_depth):
    rgb_timestamp = image_rgb.header.stamp
    rgb_info_timestamp = camera_info_rgb.header.stamp
    depth_timestamp = image_depth.header.stamp
    depth_info_timestamp = camera_info_depth.header.stamp
    wall_time = rospy.get_time()
    sim_time = rospy.Time.now()
    rospy.logerr(f"RGB: {rgb_timestamp}, RGB info: {rgb_info_timestamp}, Depth: {depth_timestamp}, Depth info: {depth_info_timestamp}")
    rospy.logerr(f"Wall time: {wall_time}, Sim time: {sim_time}")
    
    pass

def listener():
    rospy.init_node('synchronizer', anonymous=True)
    image_rgb_sub = message_filters.Subscriber('/frontleft/rgb/image_rect', Image)
    camera_info_rgb_sub = message_filters.Subscriber('/frontleft/rgb/camera_info', CameraInfo)
    image_depth_sub = message_filters.Subscriber('/frontleft/depth/image_rect', Image)
    camera_info_depth_sub = message_filters.Subscriber('/frontleft/depth/camera_info', CameraInfo)

    ts = message_filters.ApproximateTimeSynchronizer([image_rgb_sub, camera_info_rgb_sub, image_depth_sub, camera_info_depth_sub], queue_size=100, slop=1.0)
    ts.registerCallback(callback)

if __name__ == '__main__':
    listener()
    rospy.spin()
