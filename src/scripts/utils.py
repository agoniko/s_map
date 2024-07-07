from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
import rospy
import numpy as np
from functools import wraps
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import struct
import rospkg
import numpy.matlib as npm

class_names = np.loadtxt(rospkg.RosPack().get_path("s_map") + "/src/scripts/names.txt", dtype=str, delimiter=",")
label_colors = {label.lower().strip(): np.random.randint(0, 255, 3) for label in class_names}


def time_it(func):
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"time taken by {func.__name__} is {time.time()-start }")

        return result

    return wrapper


def create_pointcloud_message(objects, frame, stamp):
    """
    Creates a point cloud message from a list of points.
    """
    points = []
    for obj in objects:
        if obj.label.lower() in label_colors:
            r, g, b = label_colors[obj.label.lower()]
            a = 255
            pc = np.array(obj.points).reshape(-1, 3)
            rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, a))[0] * np.ones(
                (pc.shape[0], 1)
            )

            point = np.hstack((pc, rgb))
            points = np.vstack((points, point)) if len(points) > 0 else point
        else:
            rospy.logerr(f"Label {obj.label} not found in label colors")

    points = np.array(points, dtype=np.float32)

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 16, PointField.UINT32, 1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame
    if len(points) == 0:
        return None
    try:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        rgb = points[:, 3].astype(np.uint32)

        pc2 = point_cloud2.create_cloud(
            header, fields, [list(i) for i in zip(x, y, z, rgb)]
        )
        # pc2 = point_cloud2.create_cloud_xyz32(header, points)
        return pc2
    
    except Exception as e:
        rospy.logerr(f"Error creating point cloud message: {e}")
        return None


def create_delete_marker(frame):
    msg = MarkerArray()
    marker = Marker()
    marker.header.frame_id = frame
    marker.action = marker.DELETEALL

    msg.markers.append(marker)
    return msg


from visualization_msgs.msg import Marker, MarkerArray
import rospy
import numpy as np

def create_marker_text(label, position, marker_id, stamp, frame):
    text_marker = Marker()
    text_marker.header.frame_id = frame
    text_marker.header.stamp = stamp
    text_marker.ns = "labels"
    text_marker.id = marker_id
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose.position.x = position[0]
    text_marker.pose.position.y = position[1]
    text_marker.pose.position.z = position[2] + 0.5
    text_marker.pose.orientation.x = 0.0
    text_marker.pose.orientation.y = 0.0
    text_marker.pose.orientation.z = 0.0
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.z = 0.3  # Adjust text size as needed
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 1.0
    text_marker.color.a = 1.0
    text_marker.text = label

    return text_marker

def create_marker_array(objects, frame, stamp):
    if len(objects) == 0:
        return None, None

    msg = MarkerArray()
    label_msg = MarkerArray()

    for obj in objects:
        bbox = np.array(obj.bbox).reshape(8, 3)
        marker = create_marker_vertices(bbox, obj.label, obj.id, stamp, frame)
        if marker is not None:
            msg.markers.append(marker)
            
            # Calculate central position of the bounding box
            central_position = bbox.mean(axis=0)
            if np.sum(np.abs(central_position)) > 0:
                label_marker = create_marker_text(f"{obj.label}:{obj.id}", central_position, obj.id, stamp, frame)
                label_msg.markers.append(label_marker)
            
    if label_msg.markers:
        return (msg, label_msg)
    else:
        return (msg, None)

# Example usage
# objects should be a list of objects, each having bbox (list of 8 points), label (string), and id (int) attributes.



def create_marker_vertices(vertices, label, id, stamp, frame) -> Marker:
    """
    creates marker msg for rviz vsualization of the 3d bounding box
    """
    if vertices is None:
        return None
    marker = Marker()
    # keeping frame and timestamp consistent with the header of the received message to account for detection and mapping delay
    marker.header.stamp = stamp
    marker.header.frame_id = frame
    marker.id = int(id)

    marker.ns = "my_namespace"
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.color.a = 1.0
    marker.scale.x = 0.05
    marker.pose.orientation.w = 1.0

    connections = [
        (0, 1),  # Bottom face
        (0, 2),
        (1, 7),
        (2, 7),
        (3, 6),  # Top face
        (3, 5),
        (4, 6),
        (4, 5),
        (0, 3),  # Side faces
        (1, 6),
        (2, 5),
        (7, 4),
    ]
    for conn in connections:
        marker.points.append(Point(*vertices[conn[0]]))
        marker.points.append(Point(*vertices[conn[1]]))

    if label.lower() in label_colors:
        r, g, b = label_colors[label.lower()]
        marker.color.r = r / 255.0
        marker.color.g = g / 255.0
        marker.color.b = b / 255.0
        return marker
    else:
        return None


def get_vercitces(point_min, point_max):
    vertices = [
        #     X             Y             Z
        [point_min[0], point_min[1], point_min[2]],
        [point_max[0], point_min[1], point_min[2]],
        [point_min[0], point_max[1], point_min[2]],
        [point_max[0], point_max[1], point_min[2]],
        [point_min[0], point_min[1], point_max[2]],
        [point_max[0], point_min[1], point_max[2]],
        [point_min[0], point_max[1], point_max[2]],
        [point_max[0], point_max[1], point_max[2]],
    ]
    return vertices


def delete_marker(marker_id, frame):
    marker = Marker()
    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()
    marker.ns = "my_namespace"
    marker.id = marker_id
    marker.action = Marker.DELETE
    return marker


def bbox_iou(box1, box2):
    """
    bbox = np.array([xmin, ymin,xmax, ymax])
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def compute_3d_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two 3D boxes.

    Parameters:
    - box1: (8, 3) numpy array of vertices for the first box.
    - box2: (8, 3) numpy array of vertices for the second box.

    Returns:
    - float: the IoU of the two boxes.
    """
    # Extract the min and max points
    min_point1 = np.min(box1, axis=0)
    max_point1 = np.max(box1, axis=0)
    min_point2 = np.min(box2, axis=0)
    max_point2 = np.max(box2, axis=0)

    # Calculate intersection bounds
    inter_min = np.maximum(min_point1, min_point2)
    inter_max = np.minimum(max_point1, max_point2)
    inter_dims = np.maximum(inter_max - inter_min, 0)

    # Intersection volume
    inter_volume = np.prod(inter_dims)

    # Volumes of the individual boxes
    volume1 = np.prod(max_point1 - min_point1)
    volume2 = np.prod(max_point2 - min_point2)

    # Union volume
    union_volume = volume1 + volume2 - inter_volume

    # Intersection over Union
    iou = inter_volume / union_volume if union_volume != 0 else 0

    return iou

"""
Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." 
Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197. 
Link: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
"""
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * np.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)

def create_marker_point(points, stamp, world_frame):
    marker_array = MarkerArray()

    for i, point in enumerate(points, 1):
        marker = Marker()
        marker.header.frame_id = world_frame
        marker.header.stamp = stamp
        marker.ns = "Close_point"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker_array.markers.append(marker)

    return marker_array
