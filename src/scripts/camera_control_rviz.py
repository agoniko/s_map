#!/usr/bin/env python3
import rospy
from math import cos, sin, atan2
from view_controller_msgs.msg import CameraPlacement
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry

class CameraController:
    def __init__(self):
        rospy.init_node("camera_test", anonymous=True)
        
        self.current_X = None
        self.current_Y = None
        self.current_theta = None
        self.last_X = None
        self.last_Y = None
        self.last_theta = None

        self.pub = rospy.Publisher("/rviz/camera_placement", CameraPlacement, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.set_robot_pose)

        self.rate = rospy.Rate(30.0)
        self.interpolation_factor = 0.05  # Adjust this factor for smoother interpolation

    def set_robot_pose(self, odom):
        self.last_X = self.current_X if self.current_X is not None else odom.pose.pose.position.x
        self.last_Y = self.current_Y if self.current_Y is not None else odom.pose.pose.position.y
        self.last_theta = self.current_theta if self.current_theta is not None else 0

        self.current_X = odom.pose.pose.position.x
        self.current_Y = odom.pose.pose.position.y
        orientation_q = odom.pose.pose.orientation
        self.current_theta = atan2(2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y), 
                                   1.0 - 2.0 * (orientation_q.y ** 2 + orientation_q.z ** 2))

    def interpolate(self, start, end, factor):
        return start + factor * (end - start)

    def run(self):
        while not rospy.is_shutdown():
            if self.current_X is None or self.current_Y is None or self.current_theta is None:
                rospy.loginfo("Waiting for robot pose")
                self.rate.sleep()
                continue

            cp = CameraPlacement()

            # Interpolate positions
            X = self.interpolate(self.last_X, self.current_X, self.interpolation_factor)
            Y = self.interpolate(self.last_Y, self.current_Y, self.interpolation_factor)
            theta = self.interpolate(self.last_theta, self.current_theta, self.interpolation_factor)

            # Set the target frame to "base_link"
            cp.target_frame = "base_link"

            # Define the camera's eye position (behind the robot)
            eye_x = X + 5 * cos(theta)
            eye_y = Y + 5 * sin(theta)
            cp.eye.point = Point(eye_x, eye_y, 3)
            cp.eye.header.frame_id = "base_link"

            # Define the camera's focus point (following the robot's position and orientation)
            cp.focus.point = Point(X+1, Y, 1)
            cp.focus.header.frame_id = "base_link"

            # Define the camera's up vector (pointing upwards)
            cp.up.vector = Vector3(0, 0, 1)
            cp.up.header.frame_id = "base_link"

            # Set the time from start
            cp.time_from_start = rospy.Duration(1.0 / 30.0)

            # Publish the camera placement message
            rospy.loginfo("Publishing a camera placement message!")
            self.pub.publish(cp)

            # Sleep to maintain the desired rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        camera_controller = CameraController()
        camera_controller.run()
    except rospy.ROSInterruptException:
        pass