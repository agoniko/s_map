import numpy as np
import rospy
from collections import deque


class Obj:
    """
    Object class: Each object is characterized by 3d bboxes, label, and confidence score
    In this domain points should be stored as 8x3 array where each row is a point in the 3d space.
    """

    __slots__ = ["points", "label", "score", "last_seen", "last_checked"]
    points: np.ndarray
    label: str
    score: float
    last_seen: float
    last_checked: float

    def __init__(self, points: np.ndarray, label: str, score: float, stamp):
        assert points.shape == (8, 3)
        # Now we will sort vertices world in order to always have the same order, the keys are z then y then x
        # (useful for subsequent perception by a different pov)
        # The idea of last seen and last checked is the following:
        # Last_seen: the last time the object was detected
        # Last_checked: the last time the object was projected succesfully in the image plane
        # They are both used to check if the object is still in the world or not.
        points = points[
            np.lexsort(
                (
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                )
            )
        ]
        self.points = points
        self.label = label
        self.score = score
        self.last_seen = stamp
        self.last_checked = stamp

    def _calculate_box_area(self, vertices):
        # Calculate the area of the box deined by its vertices
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        side_lengths = max_point - min_point
        area = side_lengths[0] * side_lengths[1] * side_lengths[2]
        return area

    def _calculate_intersection_area(self, vertices1, vertices2):
        # Calculate the intersection area between two boxes
        min_point = np.maximum(np.min(vertices1, axis=0), np.min(vertices2, axis=0))
        max_point = np.minimum(np.max(vertices1, axis=0), np.max(vertices2, axis=0))
        side_lengths = np.maximum(0, max_point - min_point)
        intersection_area = side_lengths[0] * side_lengths[1] * side_lengths[2]
        return intersection_area

    def IoU(self, other):
        # Calculate the IoU between two bounding boxes
        intersection_area = self._calculate_intersection_area(self.points, other.points)
        area1 = self._calculate_box_area(self.points)
        area2 = self._calculate_box_area(other.points)
        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou


class World:
    """
    It stores objects as a dictionary of object_id -> List[Obj] because the same object can be detected multiple times.
    For this reason we need to store the history of the object in order to represent its coordinates with an estimator
    Estimator: now median maybe in the future kalman filter
    id is not a property of Obj class since it can change during time due to tracking
    """

    __slots__ = ["objects", "id2world"]

    objects: dict
    id2world: dict

    def __init__(self):
        self.objects = dict()
        self.id2world = dict()

    def exist(self, current_id: int, object: Obj, IoU_thr: float = 0.0):
        """
        based on object coordinates it returns true if the object is already in the world.
        This search is based on volume overlap. If we have a volume overlap > threshold we consider the object as the same.
        returns (true, id) if it exists, (false, -1) otherwise
        """
        for world_id, obj in self.objects.items():
            if (
                world_id != current_id
                and obj["actual"].IoU(object) > IoU_thr
                and object.label == obj["actual"].label
            ):
                return True, world_id
        return False, -1

    def remove_object(self, id: int):
        """
        Remove an object from the world
        """
        if id in self.id2world:
            world_id = self.id2world[id]
            self.objects.pop(world_id)
            self.id2world.pop(id)
            if id != world_id:
                self.id2world.pop(world_id)
        else:
            raise ValueError("Object not found in the world")

    def get_object(self, id: int, check_existence: int = 50):
        """
        Get the object with the given id
        """
        if id in self.id2world:
            w_id = self.id2world[id]
            obj = self.objects[w_id]["actual"]

            if w_id == id and len(self.objects[id]["history"]) < check_existence:
                already_exists, world_id = self.exist(id, obj)
                if already_exists:
                    self.id2world[id] = world_id
                    rospy.loginfo(
                        f"{self.objects[world_id]['actual'].label} with track id: {id} already exists in the world as {world_id}"
                    )
                    self.objects[world_id]["history"].extend(
                        self.objects[id]["history"]
                    )
                    self.update(world_id)
                    self.objects.pop(id)
                    return self.objects[world_id]["actual"], world_id

            return obj, w_id
        else:
            raise ValueError("Object not found in the world")

    def update(self, id: int) -> np.ndarray:
        """
        Update the world_id object with the median position of the history of the object i the world
        """
        if id in self.id2world:
            world_id = self.id2world[id]
            points = np.array([obj.points for obj in self.objects[world_id]["history"]])
            # extract label with highest score
            label_idx = np.argmax(
                [obj.score for obj in self.objects[world_id]["history"]]
            )
            label = self.objects[world_id]["history"][label_idx].label
            median = np.median(points, axis=0)
            # median = np.mean(points, axis=0)
            self.objects[world_id]["actual"] = Obj(
                median, label, 1.0, self.objects[world_id]["history"][-1].last_seen
            )

    def register_object(self, id, object: Obj):
        """
        Register an object in the world. If the object already exists it updates the object coordinates
        """
        if id in self.id2world:
            world_id = self.id2world[id]
            self.objects[world_id]["history"].append(object)
            self.update(world_id)
        else:
            self.objects[id] = {
                # history is a circular queue containing a maximum of 100 elements (the 100 most recent detections)
                # this avoid too much memory usage but over-rely on detections
                "history": deque([object], maxlen=100),
                "actual": object,
            }
            self.update(id)
            self.id2world[id] = id
