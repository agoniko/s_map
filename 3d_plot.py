import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def calculate_box_area(vertices):
    # Calculate the area of the box deined by its vertices
    min_point = np.min(vertices, axis=0)
    max_point = np.max(vertices, axis=0)
    side_lengths = max_point - min_point
    area = side_lengths[0] * side_lengths[1] * side_lengths[2]
    return area


def calculate_intersection_area(vertices1, vertices2):
    # Calculate the intersection area between two boxes
    min_point = np.maximum(np.min(vertices1, axis=0), np.min(vertices2, axis=0))
    max_point = np.minimum(np.max(vertices1, axis=0), np.max(vertices2, axis=0))
    side_lengths = np.maximum(0, max_point - min_point)
    intersection_area = side_lengths[0] * side_lengths[1] * side_lengths[2]
    return intersection_area


def calculate_iou(vertices1, vertices2):
    # Calculate the IoU between two bounding boxes
    intersection_area = calculate_intersection_area(vertices1, vertices2)
    area1 = calculate_box_area(vertices1)
    area2 = calculate_box_area(vertices2)
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou


ax = plt.figure().add_subplot(projection="3d")
point1 = np.array([[0, 0, 0], [1, 1, 1]])
point2 = np.array([[0, 0, 0.2], [1, 1, 0.8]])

vertices1 = [
    [point1[0][0], point1[0][1], point1[0][2]],
    [point1[0][0], point1[0][1], point1[1][2]],
    [point1[0][0], point1[1][1], point1[0][2]],
    [point1[0][0], point1[1][1], point1[1][2]],
    [point1[1][0], point1[0][1], point1[0][2]],
    [point1[1][0], point1[0][1], point1[1][2]],
    [point1[1][0], point1[1][1], point1[0][2]],
    [point1[1][0], point1[1][1], point1[1][2]],
]

vertices2 = [
    [point2[0][0], point2[0][1], point2[0][2]],
    [point2[0][0], point2[0][1], point2[1][2]],
    [point2[0][0], point2[1][1], point2[0][2]],
    [point2[0][0], point2[1][1], point2[1][2]],
    [point2[1][0], point2[0][1], point2[0][2]],
    [point2[1][0], point2[0][1], point2[1][2]],
    [point2[1][0], point2[1][1], point2[0][2]],
    [point2[1][0], point2[1][1], point2[1][2]],
]

print(calculate_iou(vertices1, vertices2))


bbox1 = point1.flatten()
bbox2 = point2.flatten()


# create linspace of points from min max of each coordinate
x1 = np.linspace(np.min(point1[:, 0]), np.max(point1[:, 0]), 10)
y1 = np.linspace(np.min(point1[:, 1]), np.max(point1[:, 1]), 10)
z1 = np.linspace(np.min(point1[:, 2]), np.max(point1[:, 2]), 10)

x2 = np.linspace(np.min(point2[:, 0]), np.max(point2[:, 0]), 10)
y2 = np.linspace(np.min(point2[:, 1]), np.max(point2[:, 1]), 10)
z2 = np.linspace(np.min(point2[:, 2]), np.max(point2[:, 2]), 10)

# combine them to obtain a cube
x1, y1, z1 = np.meshgrid(x1, y1, z1)
x2, y2, z2 = np.meshgrid(x2, y2, z2)

center1 = np.mean(point1, axis=0)
center2 = np.mean(point2, axis=0)

# plot the points
ax.scatter(x1, y1, z1)
ax.scatter(x2, y2, z2)

# axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
