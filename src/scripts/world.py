import numpy as np
import rospy


class Obj:
    """
    Object class: Each object is characterized by 3d bboxes, label, and confidence score
    In this domain points should be a 2x3 numpy array expressed in world frame coordinates.
    x1_y1_z1, x2_y2_z2
    #Due to points transformation from camera frame to world frame, we loose the min/max relationship between the points.
    #For this reason, we need to store the points as they are and calculate the min/max when needed.
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

    def IoU(self, other) -> float:
        """
        Check if the object overlaps with another object.
        Each box has shape (2, 3) where the first row is the min coordinates and the second row is the max coordinates.
        """

        # extract min and max coordinates since the points are not ordered (WORLD FRAME COORDINATES)
        xmin, ymin, zmin = np.min(self.points, axis=0)
        xmax, ymax, zmax = np.max(self.points, axis=0)

        other_xmin, other_ymin, other_zmin = np.min(other.points, axis=0)
        other_xmax, other_ymax, other_zmax = np.max(other.points, axis=0)

        x_overlap = max(0, min(xmax, other_xmax) - max(xmin, other_xmin))
        y_overlap = max(0, min(ymax, other_ymax) - max(ymin, other_ymin))
        z_overlap = max(0, min(zmax, other_zmax) - max(zmin, other_zmin))

        intersection_volume = x_overlap * y_overlap * z_overlap

        volume1 = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        volume2 = (
            (other_xmax - other_xmin)
            * (other_ymax - other_ymin)
            * (other_zmax - other_zmin)
        )

        union_volume = volume1 + volume2 - intersection_volume

        IoU = intersection_volume / union_volume if union_volume > 0 else 0.0

        return intersection_volume


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

    def exist(
        self, current_id: int, object: Obj, IoU_thr: float = 0.0
    ):
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

    def get_object(self, id: int, check_existence: int = 30):
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
                    del self.objects[id]
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

            # restoring actual coordinates as min, max coordinates
            xmin, ymin, zmin = np.min(median, axis=0)
            xmax, ymax, zmax = np.max(median, axis=0)
            bbox_3d = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])

            self.objects[world_id]["actual"] = Obj(bbox_3d, label, 1.0)

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
                "history": [object],
                "actual": object,
            }
            self.update(id)
            self.id2world[id] = id
