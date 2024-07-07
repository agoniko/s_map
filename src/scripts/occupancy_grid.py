import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

def generate_occupancy_grid(laserscan, resolution=0.1, width=100, height=100):
    """
    Generate a 2D occupancy grid from LaserScan data.
    
    :param laserscan: LaserScan message
    :param resolution: Resolution of the occupancy grid in meters per cell
    :param width: Width of the occupancy grid in cells
    :param height: Height of the occupancy grid in cells
    :return: OccupancyGrid message
    """
    # Initialize the occupancy grid
    grid = np.full((height, width), -1)  # -1 for unknown, 0 for free, 100 for occupied
    
    # Get the origin of the grid (assume the robot is at the center)
    origin_x = width // 2
    origin_y = height // 2
    
    # Iterate over the LaserScan data
    angle = laserscan.angle_min
    for range in laserscan.ranges:
        if range < laserscan.range_min or range > laserscan.range_max:
            angle += laserscan.angle_increment
            continue

        # Compute the coordinates in the map frame
        x = range * np.cos(angle)
        y = range * np.sin(angle)

        # Convert to grid coordinates
        grid_x = int((x / resolution) + origin_x)
        grid_y = int((y / resolution) + origin_y)

        # Mark the cell as occupied
        if 0 <= grid_x < width and 0 <= grid_y < height:
            grid[grid_y, grid_x] = 100

        # Mark the cells along the ray as free
        steps = int(range / resolution)
        for i in range(steps):
            free_x = int((i * np.cos(angle) / steps / resolution) + origin_x)
            free_y = int((i * np.sin(angle) / steps / resolution) + origin_y)
            if 0 <= free_x < width and 0 <= free_y < height and grid[free_y, free_x] == -1:
                grid[free_y, free_x] = 0
        
        angle += laserscan.angle_increment

    # Create the OccupancyGrid message
    occupancy_grid = OccupancyGrid()
    occupancy_grid.header = laserscan.header
    occupancy_grid.info.resolution = resolution
    occupancy_grid.info.width = width
    occupancy_grid.info.height = height
    occupancy_grid.info.origin = Pose(Point(-origin_x * resolution, -origin_y * resolution, 0), Quaternion(0, 0, 0, 1))

    # Flatten the grid and assign to data
    occupancy_grid.data = grid.flatten().tolist()

    return occupancy_grid

if __name__ == "__main__":
    rospy.init_node('occupancy_grid_generator')

    def laserscan_callback(msg):
        occupancy_grid = generate_occupancy_grid(msg)
        occupancy_grid_pub.publish(occupancy_grid)

    laserscan_sub = rospy.Subscriber('/scan', LaserScan, laserscan_callback)
    occupancy_grid_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)

    rospy.spin()
