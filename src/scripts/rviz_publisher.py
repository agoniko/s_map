#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from s_map.srv import GetAllObjects, GetAllObjectsResponse
from s_map.msg import Object, ObjectList
from std_msgs.msg import Header
import numpy as np
from utils import create_pointcloud_message, create_marker_array, create_delete_marker


class PublisherNode:
    def __init__(self):
        rospy.init_node("publisher_node", anonymous=True)
        self.init_params()
        self.init_publishers()
        self.init_services()
        self.world_objects = None

        rospy.Timer(rospy.Duration(0.1), self.publish_data)

    def init_params(self):
        self.world_frame = rospy.get_param("~world_frame", "map")
        print("World Frame: ", self.world_frame)

    def init_publishers(self):
        self.pc_pub = rospy.Publisher("/s_map/pointcloud", PointCloud2, queue_size=10)
        self.marker_pub = rospy.Publisher("/s_map/objects", MarkerArray, queue_size=10)

    def init_services(self):
        rospy.wait_for_service("get_all_objects")
        self.world_objects_client = rospy.ServiceProxy("get_all_objects", GetAllObjects)
        rospy.loginfo("World Manager Services initialized for Rviz Publisher node")


    def publish_data(self, event):
        current_time = rospy.Time.now()
        delete_all_marker = create_delete_marker(self.world_frame)
        self.marker_pub.publish(delete_all_marker)
        try:
            response = self.world_objects_client()
            self.world_objects = response.objects.objects
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        
        if self.world_objects is None:
            return
        self.publish_pointclouds(self.world_frame, current_time)
        self.publish_markers(current_time)

    def publish_pointclouds(self, frame, stamp):
        msg = create_pointcloud_message(self.world_objects, frame, stamp)
        if msg:
            self.pc_pub.publish(msg)

    def publish_markers(self, stamp):
        boxes_msg, labels_msg = create_marker_array(
            self.world_objects, self.world_frame, stamp
        )
        if boxes_msg:
            self.marker_pub.publish(boxes_msg)
        if labels_msg:
            self.marker_pub.publish(labels_msg)


if __name__ == "__main__":
    try:
        node = PublisherNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
