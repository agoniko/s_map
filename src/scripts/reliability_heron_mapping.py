#! /usr/bin/env python3


import rospy
from message_filters import TimeSynchronizer, Subscriber
from geometric_transformations import TransformHelper
from sensor_msgs.msg import Image, CameraInfo


class ReliabilityEvaluator:
    def __init__(self, source_frame, target_frame, cache_time=60.0):
        rospy.init_node("reliability_evaluator")

        self.source_frame = rospy.get_param("~source_frame", source_frame)
        self.target_frame = rospy.get_param("~target_frame", target_frame)

        self.transformer = TransformHelper(cache_time)
        self.last_transform = None
        self.last_update = None

        self.rgb_sub = Subscriber("/realsense/rgb/image_raw", Image)
        self.depth_sub = Subscriber("/realsense/aligned_depth_to_color/image_raw", Image)
        self.camera_info_sub = Subscriber("/realsense/aligned_depth_to_color/camera_info", CameraInfo)

        self.synchronizer = TimeSynchronizer([self.rgb_sub, self.depth_sub, self.camera_info_sub], queue_size=10)
        self.synchronizer.registerCallback(self.evaluate)

        self.reliable_rgb_pub = rospy.Publisher("reliable_rgb", Image, queue_size=10)
        self.reliable_depth_pub = rospy.Publisher("reliable_depth", Image, queue_size=10)
        self.reliable_camera_info_pub = rospy.Publisher("reliable_camera_info", CameraInfo, queue_size=10)

    def evaluate(self, rgb, depth, camera_info):
        rospy.logerr("Evaluating")
        stamp = rgb.header.stamp
        if self.last_transform is None:
            self.last_transform = self.transformer.lookup_transform(
                self.source_frame, self.target_frame, stamp
            )
            self.last_update = stamp
            return False

        transform = self.transformer.lookup_transform(
            self.source_frame, self.target_frame, stamp
        )
        if transform is None:
            return False

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        z = transform.transform.translation.z

        last_x = self.last_transform.transform.translation.x
        last_y = self.last_transform.transform.translation.y
        last_z = self.last_transform.transform.translation.z

        diff = abs(x - last_x) + abs(y - last_y) + abs(z - last_z)
        self.last_transform = transform
        # considering 20 FPS as the running of the detection node, 0.2 correspond to a maximum linear velocity of 4 m/s (about 15km/h)
        if diff > 0.05:
            return 
        else:
            self.reliable_rgb_pub.publish(rgb)
            self.reliable_depth_pub.publish(depth)
            self.reliable_camera_info_pub.publish(camera_info)

if __name__ == "__main__":
    rospy.logerr("Reliability evaluator node started")
    evaluator = ReliabilityEvaluator("base_link", "camera_link")
    rospy.spin()


