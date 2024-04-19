import rospy
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2 as cv
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
import numpy as np
import tf2_ros


class CameraPoseEstimator:

    @staticmethod
    def receive_camera_info(info_topic):
        class Dummy:
            def __init__(self):
                self.data = None

            def callback(self, x):
                self.data = x
                self.camera = PinholeCameraModel()
                self.camera.fromCameraInfo(x)

        data = Dummy()
        sub = rospy.Subscriber(info_topic, CameraInfo, callback=data.callback)

        r = rospy.Rate(10)
        while data.data is None:
            r.sleep()
        sub.unregister()

        k = np.array(data.data.K).reshape((3, 3))
        height, width = data.data.height, data.data.width

        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]
        return (fx, fy), (cx, cy), (height, width), data.camera

    def __init__(self, info_topic):
        (self.fx, self.fy), (self.cx, self.cy), (self.height, self.width), self.camera = (
            self.receive_camera_info(info_topic)
        )

    def pixel_to_3d(self, pix_x, pix_y, depth_m):
        x_3d = (pix_x - self.cx) * depth_m / self.fx
        y_3d = (pix_y - self.cy) * depth_m / self.fy
        return x_3d, y_3d

    def d3_to_pixel(self, x, y, z):
        return self.camera.project3dToPixel((x, y, z))

    def normalized_pixel_to_3d(self, pix_x, pix_y, depth_m):
        pix_x = pix_x * self.width
        pix_y = pix_y * self.height
        x_3d = (pix_x - self.cx) * depth_m / self.fx
        y_3d = (pix_y - self.cy) * depth_m / self.fy
        return x_3d, y_3d


class Transformer:
    def __init__(self):
        # Initialize a TF buffer with a longer cache time
        tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(30.0)
        )  # Adjust cache time as needed (e.g., 30 seconds)

        # Initialize a TF listener
        self.tf_listener = tf2_ros.TransformListener(tf_buffer)

    def transform_coordinates(self, x, y, z, from_frame, to_frame, timestamp=None):
        stamped = PoseStamped()
        stamped.pose.position.x = x
        stamped.pose.position.y = y
        stamped.pose.position.z = z
        stamped.header.frame_id = from_frame

        if timestamp is None:
            timestamp = rospy.Time(0)

        stamped.header.stamp = timestamp

        # self.tflistener.waitForTransform(
        #    to_frame, from_frame, timestamp, rospy.Duration(5)
        # )
        self.tflistener.lookupTransform(to_frame, from_frame, timestamp)
        transformed = self.tflistener.transformPose(to_frame, stamped)

        res_x = transformed.pose.position.x
        res_y = transformed.pose.position.y
        res_z = transformed.pose.position.z
        return res_x, res_y, res_z


from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


class TransformHelper:
    def __init__(self, cache_time=60.0):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(cache_time))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def lookup_transform_and_transform_coordinates(
        self, source_frame, target_frame, point_source, stamp
    ):
        """
        point source is a list [x, y, z]
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Failed to lookup transform: %s", str(e))
            return None

        point_source_stamped = PointStamped()
        point_source_stamped.header.frame_id = source_frame
        point_source_stamped.header.stamp = stamp
        point_source_stamped.point.x = point_source[0]
        point_source_stamped.point.y = point_source[1]
        point_source_stamped.point.z = point_source[2]

        try:
            point_target = do_transform_point(point_source_stamped, transform)
            x = point_target.point.x
            y = point_target.point.y
            z = point_target.point.z

            return x, y, z
        except Exception as e:
            rospy.logerr("Failed to transform coordinates: %s", str(e))
            return None
