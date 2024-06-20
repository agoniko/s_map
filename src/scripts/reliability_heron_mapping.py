#! /usr/bin/env python3

import rospy
from message_filters import TimeSynchronizer, Subscriber
from geometric_transformations import TransformHelper
from nav_msgs.msg import Odometry
import numpy as np

class ReliabilityEvaluator:
    def __init__(self, cache_time=60.0):
        rospy.init_node("reliability_evaluator")

        self.source_frame = rospy.get_param("~source_frame")
        self.target_frame = rospy.get_param("~target_frame")

        self.transformer = TransformHelper(cache_time)
        self.last_transform = None
        self.last_update = None

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.evaluate)

        self.reliable_odom_pub = rospy.Publisher("/reliable/odom", Odometry, queue_size=10)
        self.distances = []
    
    def distance(self, transform1, transform2):
        """
        calculates the distance between two transforms
        """
        x1 = transform1.translation.x
        y1 = transform1.translation.y
        z1 = transform1.translation.z

        x2 = transform2.translation.x
        y2 = transform2.translation.y
        z2 = transform2.translation.z

        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5

    def evaluate(self, odom):
        """
        determines if the odom is reliable based on the distance with the last reliable odom
        """
        if self.last_transform is None:
            self.last_transform = self.transformer.lookup_transform(self.source_frame, self.target_frame, odom.header.stamp).transform
            self.last_update = odom.header.stamp
            return False

        current_transform = self.transformer.lookup_transform(self.source_frame, self.target_frame, odom.header.stamp).transform
        distance = self.distance(self.last_transform, current_transform)
        self.distances.append(distance)
        #rospy.logerr(np.mean(self.distances))
        #rospy.logerr(np.median(self.distances))
        #rospy.logerr(np.std(self.distances))
        #rospy.logerr(np.max(self.distances))
        #rospy.logerr(np.min(self.distances))
        if distance < 0.5 or self.last_update < odom.header.stamp - rospy.Duration(1.0):
            self.last_update = odom.header.stamp
            self.last_transform = current_transform
            self.reliable_odom_pub.publish(odom)
        else:
            rospy.logwarn("Unreliable odom detected")


if __name__ == "__main__":
    rospy.logerr("Reliability evaluator node started")
    evaluator = ReliabilityEvaluator()
    rospy.spin()






