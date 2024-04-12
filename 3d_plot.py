import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_overlap_volume(box1, box2):
    """
    each box has shape (2, 3) where the first row is the min coordinates and the second row is the max coordinates
    """
    x_overlap = max(0, min(box1[1, 0], box2[1, 0]) - max(box1[0, 0], box2[0, 0]))
    y_overlap = max(0, min(box1[1, 1], box2[1, 1]) - max(box1[0, 1], box2[0, 1]))
    z_overlap = max(0, min(box1[1, 2], box2[1, 2]) - max(box1[0, 2], box2[0, 2]))
    intersection_volume = x_overlap * y_overlap * z_overlap

    return intersection_volume


ax = plt.figure().add_subplot(projection='3d')
point1 = np.array([[0, 0, 0], [0.7, 0.5000001, 0.7]])
point2 = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])

#create linspace of points from min max of each coordinate
x1 = np.linspace(np.min(point1[:, 0]), np.max(point1[:, 0]), 10)
y1 = np.linspace(np.min(point1[:, 1]), np.max(point1[:, 1]), 10)
z1 = np.linspace(np.min(point1[:, 2]), np.max(point1[:, 2]), 10)

x2 = np.linspace(np.min(point2[:, 0]), np.max(point2[:, 0]), 10)
y2 = np.linspace(np.min(point2[:, 1]), np.max(point2[:, 1]), 10)
z2 = np.linspace(np.min(point2[:, 2]), np.max(point2[:, 2]), 10)

#combine them to obtain a cube
x1, y1, z1 = np.meshgrid(x1, y1, z1)
x2, y2, z2 = np.meshgrid(x2, y2, z2)

#plot the points
ax.scatter(x1, y1, z1)
ax.scatter(x2, y2, z2)

#axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

print(calculate_overlap_volume(point1, point2))

plt.show()
