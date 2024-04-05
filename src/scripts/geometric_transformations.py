import rospy
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2 as cv
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


class CameraPoseEstimator:

    @staticmethod
    def receive_camera_info(info_topic):
        class Dummy:
            def __init__(self):
                self.data = None
            def callback(self, x):
                self.data = x

        data = Dummy()
        sub = rospy.Subscriber(info_topic, CameraInfo, callback=data.callback)

        r = rospy.Rate(10)
        while data.data is None: r.sleep()
        sub.unregister()

        k = np.array(data.data.K).reshape((3,3))

        fx, fy = k[0,0], k[1,1]
        cx, cy = k[0,2], k[1,2]
        return (fx, fy), (cx, cy)

    def __init__(self, info_topic):
        (self.fx, self.fy), (self.cx, self.cy) = self.receive_camera_info(info_topic)

    def pixel_to_3d(self, pix_x, pix_y, depth_m):
        x_3d = (pix_x - self.cx) * depth_m / self.fx
        y_3d = (pix_y - self.cy) * depth_m / self.fy
        return x_3d, y_3d

class Transformer:
    def __init__(self):
        self.tflistener = tf.TransformListener()

    def transform_coordinates(self, x, y, z, from_frame, to_frame):
        stamped = PoseStamped()
        stamped.pose.position.x = x
        stamped.pose.position.y = y
        stamped.pose.position.z = z
        stamped.header.frame_id = from_frame

        timestamp = rospy.Time(0)
        stamped.header.stamp = timestamp

        self.tflistener.waitForTransform(to_frame, from_frame, timestamp, rospy.Duration(5))
        transformed = self.tflistener.transformPose(to_frame, stamped)

        res_x = transformed.pose.position.x
        res_y = transformed.pose.position.y
        res_z = transformed.pose.position.z
        return res_x, res_y, res_z
