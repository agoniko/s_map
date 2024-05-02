from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
import rospy


def create_marker(point_min, point_max, label, id, header) -> Marker:
    marker = Marker()
    # keeping frame and timestamp consistent with the header of the received message to account for detection and mapping delay
    marker.header = header
    marker.id = id
    marker.ns = "my_namespace"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0

    marker.color.a = 1.0

    marker.pose.position.x = (point_min[0] + point_max[0]) / 2
    marker.pose.position.y = (point_min[1] + point_max[1]) / 2
    marker.pose.position.z = (point_min[2] + point_max[2]) / 2
    marker.scale.x = point_max[0] - point_min[0]
    marker.scale.y = point_max[1] - point_min[1]
    marker.scale.z = point_max[2] - point_min[2]

    marker.pose.orientation.w = 0

    if label.lower() == "person":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    elif label.lower() == "chair":
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "laptop":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    elif label.lower() == "dining table":
        # yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "tv":
        # aqua
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
    else:
        return None

    return marker


def create_marker_vertices(vertices, label, id, stamp, frame) -> Marker:
    """
    creates marker msg for rviz vsualization of the 3d bounding box
    """
    marker = Marker()
    # keeping frame and timestamp consistent with the header of the received message to account for detection and mapping delay
    marker.header.stamp = stamp
    marker.header.frame_id = frame
    marker.id = id

    marker.ns = "my_namespace"
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.color.a = 1.0
    marker.scale.x = 0.05
    marker.pose.orientation.w = 1.0

    # create the lines of the bounding box
    # bottom face
    marker.points.append(Point(*vertices[0]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[0]))

    # top face
    marker.points.append(Point(*vertices[4]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[7]))
    marker.points.append(Point(*vertices[7]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[4]))

    # vertical lines
    marker.points.append(Point(*vertices[0]))
    marker.points.append(Point(*vertices[4]))
    marker.points.append(Point(*vertices[1]))
    marker.points.append(Point(*vertices[5]))
    marker.points.append(Point(*vertices[2]))
    marker.points.append(Point(*vertices[6]))
    marker.points.append(Point(*vertices[3]))
    marker.points.append(Point(*vertices[7]))

    if label.lower() == "person":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
    elif label.lower() == "chair":
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "laptop":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    elif label.lower() == "dining table":
        # yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif label.lower() == "tv":
        # aqua
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
    else:
        return None

    return marker


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
