#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, LaserScan
import numpy as np
import pclpy
from pclpy import pcl

def point_cloud_callback(msg):
    # Convert ROS PointCloud2 message to PCL point cloud
    point_cloud = pcl.PointCloud.PointXYZ()
    pclpy.pcl.fromROSMsg(msg, point_cloud)

    # Create a passthrough filter to get points at z = 0.5 ± 0.1
    passthrough = point_cloud.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0.4, 0.6)
    filtered_cloud = passthrough.filter()

    # Convert filtered point cloud to a numpy array
    points = np.array(filtered_cloud.to_array())

    # Convert points to LaserScan message
    laser_scan = convert_to_laserscan(points)

    # Publish the LaserScan message
    laser_scan_pub.publish(laser_scan)

def convert_to_laserscan(points):
    # Create a LaserScan message
    laser_scan = LaserScan()
    laser_scan.header.stamp = rospy.Time.now()
    laser_scan.header.frame_id = "mir/odom"
    laser_scan.angle_min = -np.pi
    laser_scan.angle_max = np.pi
    laser_scan.angle_increment = np.pi / 180.0  # 1 degree resolution
    laser_scan.range_min = 0.0
    laser_scan.range_max = 100.0  # Set according to your needs

    # Initialize ranges with infinity
    num_readings = int((laser_scan.angle_max - laser_scan.angle_min) / laser_scan.angle_increment)
    ranges = np.full(num_readings, np.inf)

    # Populate the ranges
    for point in points:
        x, y = point[0], point[1]
        angle = np.arctan2(y, x)
        range_distance = np.sqrt(x**2 + y**2)
        if laser_scan.angle_min <= angle <= laser_scan.angle_max:
            index = int((angle - laser_scan.angle_min) / laser_scan.angle_increment)
            if range_distance < ranges[index]:
                ranges[index] = range_distance

    laser_scan.ranges = ranges.tolist()
    return laser_scan

if __name__ == "__main__":
    rospy.init_node("cloud_to_laserscan")

    # Subscriber to the point cloud topic
    rospy.Subscriber("/rtabmap/cloud_map", PointCloud2, point_cloud_callback)

    # Publisher for the LaserScan message
    laser_scan_pub = rospy.Publisher("/converted_laserscan", LaserScan, queue_size=10)

    rospy.spin()
