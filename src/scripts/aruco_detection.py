#! /usr/bin/env python3

import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometric_transformations import CameraPoseEstimator
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
import tf.transformations as tf_trans
import numpy as np
import tf

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector', anonymous=True)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/realsense/rgb/image_raw', Image, self.image_callback)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Define ArUco dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Camera pose estimator
        self.camera_pose_estimator = CameraPoseEstimator('/realsense/rgb/camera_info')
        
        self.camera_matrix = self.camera_pose_estimator.camera.K
        self.dist_coeffs = self.camera_pose_estimator.camera.D

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Detect ArUco markers
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        if ids is not None:
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, self.camera_matrix, self.dist_coeffs)
                self.publish_tf(ids[i][0], rvec[0][0], tvec[0][0], data.header.stamp)

    def publish_tf(self, marker_id, rvec, tvec, stamp):
        try:
            # Lookup the transform from map to realsense_rgb_optical_frame
            transform = self.tf_buffer.lookup_transform(
                "mir/odom", "realsense_rgb_optical_frame", stamp, timeout=rospy.Duration(5)
            )
        except tf2_ros.TransformException as ex:
            rospy.logerr(f"Failed to lookup transform: {ex}")
            return

        # Convert rvec to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(np.array(rvec))
        
        # Construct the 4x4 rototranslation matrix from rvec and tvec
        rototranslation_matrix = np.eye(4)
        rototranslation_matrix[:3, :3] = rotation_matrix
        rototranslation_matrix[:3, 3] = np.array(tvec)

        # Extract the translation and rotation from the transform
        map_to_optical_translation = [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ]
        map_to_optical_rotation = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ]

        # Convert the transform to a 4x4 matrix
        map_to_optical_matrix = tf.transformations.quaternion_matrix(map_to_optical_rotation)
        map_to_optical_matrix[:3, 3] = map_to_optical_translation

        # Combine the transforms
        combined_matrix = np.dot(map_to_optical_matrix, rototranslation_matrix)

        # Create a TransformStamped message
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = stamp
        transform_stamped.header.frame_id = "map"
        transform_stamped.child_frame_id = f"marker_{marker_id}"
        
        # Fill the translation
        transform_stamped.transform.translation.x = combined_matrix[0, 3]
        transform_stamped.transform.translation.y = combined_matrix[1, 3]
        transform_stamped.transform.translation.z = combined_matrix[2, 3]
        
        # Fill the rotation (convert rotation matrix to quaternion)
        combined_quaternion = tf.transformations.quaternion_from_matrix(combined_matrix)
        transform_stamped.transform.rotation.x = combined_quaternion[0]
        transform_stamped.transform.rotation.y = combined_quaternion[1]
        transform_stamped.transform.rotation.z = combined_quaternion[2]
        transform_stamped.transform.rotation.w = combined_quaternion[3]
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_stamped)
            

if __name__ == '__main__':
    try:
        ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()