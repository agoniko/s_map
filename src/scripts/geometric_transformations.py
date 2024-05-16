import rospy
from cv_bridge import CvBridge, CvBridgeError
import tf
import cv2 as cv
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel
import numpy as np
import tf2_ros

from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
import tf.transformations as tft
import math


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
        (
            (self.fx, self.fy),
            (self.cx, self.cy),
            (self.height, self.width),
            self.camera,
        ) = self.receive_camera_info(info_topic)

    def pixel_to_3d(self, pix_x, pix_y, depth_m):
        # getting the unit vector that represent the 3d ray from pixel (pix_x, pix_y) = (u, v)
        (x, y, z) = self.camera.projectPixelTo3dRay((pix_x, pix_y))

        # scaling the unit vector by the depth to get the 3d point
        x_3d = x * depth_m
        y_3d = y * depth_m
        z_3d = z * depth_m

        return [x_3d, y_3d, z_3d]

    def multiple_pixels_to_3d(self, uvz):
        """
        Parallelized version of pixel_to_3d for mutliple points modified from the official library
        :param uvz:        np.ndarray of rectified pixel coordinates + depth estimation in meters
        :type uvz:         [(u, v, z), ...]

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        u = (uvz[:, 0] - self.camera.cx()) / self.camera.fx()
        v = (uvz[:, 1] - self.camera.cy()) / self.camera.fy()
        depth = uvz[:, 2]

        # extracting unit 3dray
        uv = np.stack((u, v, np.ones(len(u))), axis=1)
        norms = np.linalg.norm(uv, axis=1, keepdims=False)

        u /= norms
        v /= norms
        z = np.ones(len(norms)) / norms

        # scaling the unit vector by the depth to get the 3d point
        x_3d = u * depth
        y_3d = v * depth
        z_3d = z * depth

        points = np.stack((x_3d, y_3d, z_3d), axis=1)
        return points

    def d3_to_pixel(self, x, y, z):
        return self.camera.project3dToPixel((x, y, z))

    def rectify_image(self, image):
        rect = np.zeros_like(image)
        self.camera.rectifyImage(image, rect)
        return rect


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class TransformHelper(metaclass=SingletonMeta):
    def __init__(self, cache_time=60.0):
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(cache_time))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    # this function return the translation and rotation of a frame without performing any transformation
    def lookup_transform(self, source_frame, target_frame, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp, timeout=rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr("Failed to lookup transform: %s", str(e))
            return None
        
        return transform

    def transform_coordinates(self, source_frame, target_frame, points, stamp):
        """
        point source is a list [[x, y, z], [x, y, z], ...]
        """
        transform = self.lookup_transform(source_frame, target_frame, stamp)
        if transform is None:
            return None

        transformed_points = []
        for point in points:
            point_source_stamped = PointStamped()
            point_source_stamped.header.frame_id = source_frame
            point_source_stamped.header.stamp = stamp
            point_source_stamped.point.x = point[0]
            point_source_stamped.point.y = point[1]
            point_source_stamped.point.z = point[2]

            try:
                point_target = do_transform_point(point_source_stamped, transform)
                x = point_target.point.x
                y = point_target.point.y
                z = point_target.point.z

                transformed_points.append([x, y, z])
            except Exception as e:
                rospy.logerr("Failed to transform coordinates: %s", str(e))
                return None

        return np.array(transformed_points)

    def transform_points(self, points, rotation, translation):
        """
        Points has shape n, 3.
        This function creates the rototranslation matrix from the rotation and translation and applies it to the points.
        """

        # Create the rototranslation matrix
        rototranslation = tft.quaternion_matrix(rotation)
        rototranslation[:3, 3] = translation

        # Apply the rototranslation matrix to the points
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        points = np.dot(rototranslation, points.T).T

        return points[:, :3]

    def fast_transform(self, source_frame, target_frame, points, stamp):
        transform = self.lookup_transform(source_frame, target_frame, stamp)
        if transform is None:
            return None

        rotation = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]
        translation = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]

        return self.transform_points(points, rotation, translation)
