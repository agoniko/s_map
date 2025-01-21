#!/Users/nicoloagostara/miniforge3/envs/ros_env/bin/python3


import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import rospkg

def convert_open3d_to_ros(point_cloud):
    """Convert an Open3D point cloud to ROS PointCloud2."""
    # Extract point data from Open3D
    points = point_cloud.points
    colors = point_cloud.colors

    # Combine the point and color data
    points_with_color = []
    for i in range(len(points)):
        x, y, z = points[i]
        r, g, b = (colors[i] * 255).astype(int)  # RGB values in Open3D are between 0 and 1
        rgb = (r << 16) | (g << 8) | b  # Combine RGB into a single integer
        points_with_color.append([x, y, z, rgb])

    # Create PointCloud2 message
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "mir/odom"  # Set the reference frame, adjust as needed

    # Create PointCloud2 message using pc2.create_cloud
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    point_cloud_msg = pc2.create_cloud(header, fields, points_with_color)

    return point_cloud_msg

def publish_ply_to_pointcloud2(ply_file_path):
    # Load the .ply file using Open3D
    pcd = o3d.io.read_point_cloud(ply_file_path)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(pcd.points) == 0:
        rospy.logerr("No points loaded from the .ply file")
        return

    rospy.loginfo(f"Loaded {len(pcd.points)} points from the .ply file.")

    # Initialize ROS node and publisher
    rospy.init_node('ply_to_pointcloud2_publisher', anonymous=True)
    pub = rospy.Publisher('/pointcloud2_topic', PointCloud2, queue_size=10)
    pointcloud_msg = convert_open3d_to_ros(pcd)
    rate = rospy.Rate(1)  # Publish at 1 Hz
    while not rospy.is_shutdown():
        # Convert the Open3D point cloud to ROS PointCloud2 message
        pub.publish(pointcloud_msg)
        rospy.loginfo("Published point cloud to /pointcloud2_topic")
        rate.sleep()

if __name__ == "__main__":
    ply_file_path = rospkg.RosPack().get_path("s_map") + "/labelCloud/pointclouds/cloud_scene7.ply"  # Specify the path to your .ply file
    publish_ply_to_pointcloud2(ply_file_path)