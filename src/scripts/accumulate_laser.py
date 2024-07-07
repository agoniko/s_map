#! /usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import Point32
import open3d as o3d
import tf
from geometric_transformations import TransformHelper
import rospkg

SAVE_PATH = rospkg.RosPack().get_path("s_map") + "/pointclouds/notebooks/map_evaluation/"

class LaserScanToPointCloud:
    def __init__(self):
        rospy.init_node("laserscan_to_pointcloud")
        self.tf_helper = TransformHelper()

        self.world_frame = rospy.get_param("~world_frame", "map")
        self.combined_points = []
        self.last_rtabmap_cloud = None

        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)
        rospy.Subscriber("/write_on_file", Bool, self.write_on_file_callback)
        rospy.Subscriber("/rtabmap/cloud_map", PointCloud2, self.point_cloud_callback)

    def point_cloud_callback(self, msg):
        self.last_rtabmap_cloud = msg

    def write_on_file_callback(self, msg):
        if msg.data:
            self.save_point_cloud(SAVE_PATH + "accumulated_laserscan.ply", self.combined_points)

            if self.last_rtabmap_cloud is not None:
                rtabmap_points = list(point_cloud2.read_points(
                    self.last_rtabmap_cloud,
                    field_names=("x", "y", "z"),
                    skip_nans=True
                ))
                self.save_point_cloud(SAVE_PATH + "rtabmap_pointcloud.ply", rtabmap_points)

    def save_point_cloud(self, filename, points):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pc)

    def laserscan_callback(self, msg):
        points = self.laserscan_to_pointcloud(msg)
        points_world_frame = self.tf_helper.fast_transform(msg.header.frame_id, self.world_frame, points, msg.header.stamp)
        self.combined_points.extend(points_world_frame)

    def laserscan_to_pointcloud(self, scan):
        points = []
        angle = scan.angle_min

        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0  # Assuming a 2D laser scanner
                points.append((x, y, z))

            angle += scan.angle_increment

        return points

if __name__ == "__main__":
    LaserScanToPointCloud()
    rospy.spin()
