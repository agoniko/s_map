#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from utils import time_it
from geometric_transformations import TransformHelper

class PointCloudMerger:
    def __init__(self):
        rospy.init_node('pointcloud_merger')
        self.subscribers = []
        topics = rospy.get_param('~topics')
        self.world_frame = rospy.get_param('~world_frame')
        self.pub = rospy.Publisher('/merged_pointcloud', PointCloud2, queue_size=10)
        self.transformer = TransformHelper()

        for topic in topics.split(' '):
            self.subscribers.append(Subscriber(topic, PointCloud2))

        self.ats = ApproximateTimeSynchronizer(self.subscribers, queue_size=10, slop=0.5)
        self.ats.registerCallback(self.callback)

    def callback(self, *clouds):
        merged_points = []
        
        for cloud in clouds:
            points = np.array(list(pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z", "rgb"))))
            points_world_frame = self.transformer.fast_transform(cloud.header.frame_id, self.world_frame, points[:, :3], cloud.header.stamp)
            merged_points.extend(np.hstack((points_world_frame, points[:, 3:4])))

        merged_cloud = self.convert_to_pointcloud2(merged_points)
        merged_cloud.header.stamp = rospy.Time.now()
        merged_cloud.header.frame_id = self.world_frame
        self.pub.publish(merged_cloud)

    def convert_to_pointcloud2(self, points):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'vision'
        
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1)
        ]
        return pc2.create_cloud(header, fields, points)

if __name__ == '__main__':
    try:
        merger = PointCloudMerger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
