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
            pc = np.asarray(obj.pcd.points)
            rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, a))[0] * np.ones(
                (pc.shape[0], 1)
            )

            point = np.hstack((pc, rgb))
            points = np.vstack((points, point)) if len(points) > 0 else point
        else:
            rospy.logerr(f"Label {obj.label} not found in label colors")

    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("rgb", 16, PointField.UINT32, 1),
    ]

    header = Header()
    header.stamp = stamp
    header.frame_id = frame

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    rgb = points[:, 3].astype(np.uint32)

    pc2 = point_cloud2.create_cloud(
        header, fields, [list(i) for i in zip(x, y, z, rgb)]
    )
    # pc2 = point_cloud2.create_cloud_xyz32(header, points)
    return pc2


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
        return None

    msg = MarkerArray()
    label_msg = MarkerArray()

    for obj in objects:
        marker = create_marker_vertices(obj.bbox, obj.label, obj.id, stamp, frame)
        if marker is not None:
            msg.markers.append(marker)
            
            # Calculate central position of the bounding box
            bbox = np.array(obj.bbox)
            central_position = bbox.mean(axis=0)
            if np.sum(np.abs(central_position)) > 0:
                label_marker = create_marker_text(obj.label, central_position, obj.id, stamp, frame)
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
