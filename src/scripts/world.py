import numpy as np
import rospy


class Obj:
    """
    Object class: Each object is characterized by 3d bboxes, label, and confidence score
    In this domain points should be a 2x3 numpy array expressed in world frame coordinates.
    xmin_ymin_zmin, xmax_ymax_zmax
    """

    __slots__ = ["points", "label", "score"]
    points: np.ndarray
    label: str
    score: float

    def __init__(self, points: np.ndarray, label: str, score: float):
        assert points.shape == (2, 3)
        self.points = points
        self.label = label
        self.score = score

    def overlap_volume(self, other) -> float:
        """
        Check if the object overlaps with another object.
        Each box has shape (2, 3) where the first row is the min coordinates and the second row is the max coordinates.
        NOTE: The output volume has the same unit of measure of the input data (in our use case m3)
        TODO: This function should return a value between 0 and 1
        IDEA: Implement IoU for 3d objects
        """
        x_overlap = max(
            0,
            min(self.points[1, 0], other.points[1, 0])
            - max(self.points[0, 0], other.points[0, 0]),
        )
        y_overlap = max(
            0,
            min(self.points[1, 1], other.points[1, 1])
            - max(self.points[0, 1], other.points[0, 1]),
        )
        z_overlap = max(
            0,
            min(self.points[1, 2], other.points[1, 2])
            - max(self.points[0, 2], other.points[0, 2]),
        )

        intersection_volume = x_overlap * y_overlap * z_overlap

        return intersection_volume


class World:
    """
    It stores objects as a dictionary of object_id -> List[Obj] because the same object can be detected multiple times.
    For this reason we need to store the history of the object in order to represent its coordinates with an estimator
    Estimator: now median maybe in the future kalman filter
    id is not a property of Obj class since it can change during time due to tracking
    """

    __slots__ = ["objects"]

    objects: dict

    def __init__(self):
        self.objects = dict()

    def exist(self, current_id: int, object: Obj) -> tuple[bool, int]:
        """
        based on object coordinates it returns true if the object is already in the world.
        This search is based on volume overlap. If we have a volume overlap > threshold we consider the object as the same.
        returns (true, id) if it exists, (false, -1) otherwise
        """
        for world_id, obj in self.objects.items():
            if (
                world_id != current_id
                and obj["actual"].overlap_volume(object) > 0.0
            ):
                return True, world_id
        return False, -1

    def get_object(self, id: int) -> Obj:
        """
        Get the object with the given id
        """
        if id in self.objects:
            obj = self.objects[id]["actual"]
            already_exists, world_id = self.exist(id, obj)
            #never goes here
            if already_exists:
                rospy.loginfo(f"Object {id} already exists in the world as {world_id}")
                self.objects[world_id]["history"].append(self.objects[id]["history"])
                self.update(world_id)
                self.objects[id] = self.objects[world_id]
                return self.objects[world_id]["actual"], world_id
            else:
                return obj, id
        else:
            raise ValueError("Object not found in the world")

    def update(self, id: int) -> np.ndarray:
        """
        Update the world_id object with the median position of the history of the object i the world
        """
        if id in self.objects:
            points = np.array([obj.points for obj in self.objects[id]["history"]])
            # extract label with highest score
            label_idx = np.argmax([obj.score for obj in self.objects[id]["history"]])
            label = self.objects[id]["history"][label_idx].label
            median = np.median(points, axis=0)
            self.objects[id]["actual"] = Obj(median, label, 1.0)

    def register_object(self, id, object: Obj):
        """
        Register an object in the world. If the object already exists it updates the object coordinates
        """
        if id in self.objects:
            self.objects[id]["history"].append(object)
            self.update(id)
        else:
            self.objects[id] = {
                "history": [object],
                "actual": object,
            }
