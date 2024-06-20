
# ros packages
import rospy

# others
from geometric_transformations import TransformHelper


class ReliabilityEvaluator:
    def __init__(self, source_frame, target_frame, cache_time=60.0):
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.transformer = TransformHelper(cache_time)
        self.last_transform = None
        self.last_update = None

    def evaluate(self, stamp):
        if self.last_transform is None:
            self.last_transform = self.transformer.lookup_transform(
                self.source_frame, self.target_frame, stamp
            )
            self.last_update = stamp
            return False

        transform = self.transformer.lookup_transform(
            self.source_frame, self.target_frame, stamp
        )
        if transform is None:
            return False

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        z = transform.transform.translation.z

        last_x = self.last_transform.transform.translation.x
        last_y = self.last_transform.transform.translation.y
        last_z = self.last_transform.transform.translation.z

        diff = abs(x - last_x) + abs(y - last_y) + abs(z - last_z)
        self.last_transform = transform
        # considering 20 FPS as the running of the detection node, 0.2 correspond to a maximum linear velocity of 4 m/s (about 15km/h)
        if diff > 0.05:
            return False
        else:
            return True
