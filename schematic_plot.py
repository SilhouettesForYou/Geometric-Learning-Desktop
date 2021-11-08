import matplotlib.pyplot as plt
import numpy as np
import os
import vtk

from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def read_stl(file_name):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_name)
    reader.Update()

    poly_data = reader.GetOutput()
    return poly_data

def serialize_centers(centers):
    k = len(centers)
    kdtree = KDTree(centers, leaf_size=30, metric='euclidean')
    distances, mapping = kdtree.query(centers, k=k, return_distance=True)
    
    start = np.where(distances[:, k - 1] == max(distances[:, k - 1]))
    start = start[0][0]
    end = mapping[start][k - 1]

    serialized_indices = [start]
    while start != end:
        i = 1
        while mapping[start][i] in serialized_indices:
            i += 1
        start = mapping[start][i]
        serialized_indices.append(start)

    return np.array([centers[x] for x in serialized_indices])

def update_position(e,fig,ax,labels_and_points):
    for label, x, y, z in labels_and_points:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        label.xy = x2,y2
        label.update_positions(fig.canvas.renderer)
    fig.canvas.draw()


def plot(fig, points, color, position, title, labels, cluster=6):
    ax = fig.add_subplot(position, projection = '3d')
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_
    center = np.mean(centers, 0)

    pca = PCA(n_components=3)
    pca.fit(np.array(centers))

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o', s=50, c=color)

    vecs = np.array([pca.components_[0], pca.components_[1], -pca.components_[0], -pca.components_[1]])
    points_ = np.zeros(vecs.shape)
    for i in range(len(vecs)):
        if i & 1:
            points_[i] = center + vecs[i] * 10
        else:
            points_[i] = center + vecs[i] * 50
    plotlabels = []
    xs, ys, zs = np.split(points_, 3, axis=1)
    sc = ax.scatter(xs,ys,zs)
    el = Ellipse((2, -1), 0.5, 0.5)
    for txt, x, y, z in zip(labels, xs, ys, zs):
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        label = plt.annotate(
            txt, xy = (x2, y2), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'darkkhaki', alpha = 1.0),
            arrowprops=dict(arrowstyle="wedge,tail_width=1.",
                                  fc='darkkhaki', ec="none",
                                  patchA=None,
                                  patchB=el,
                                  relpos=(0.2, 0.5)))
        plotlabels.append(label)
    fig.canvas.mpl_connect('motion_notify_event', lambda event: update_position(event, fig, ax, zip(plotlabels, xs, ys, zs)))
    
    # draw main axis
    x, y, z = np.meshgrid(np.array([center[0] for i in range(2)]), \
         np.array([center[1] for i in range(2)]), np.array([center[2] for i in range(2)]))
    u = np.array([pca.components_[0][0], -pca.components_[0][0]])
    v = np.array([pca.components_[0][1], -pca.components_[0][1]])
    w = np.array([pca.components_[0][2], -pca.components_[0][2]])
    ax.quiver(x, y, z, u, v, w, length=50)

    # draw secondary axis
    x, y, z = np.meshgrid(np.array([center[0] for i in range(2)]), \
         np.array([center[1] for i in range(2)]), np.array([center[2] for i in range(2)]))
    u = np.array([pca.components_[1][0], -pca.components_[1][0]])
    v = np.array([pca.components_[1][1], -pca.components_[1][1]])
    w = np.array([pca.components_[1][2], -pca.components_[1][2]])
    ax.quiver(x, y, z, u, v, w, length=10)

    # ax.set_title(title)

fixed_poly_data = read_stl('./fractures/3-2-a.stl')
float_poly_data = read_stl('./fractures/3-3-a.stl')

fixed_poly_points = fixed_poly_data.GetPoints()
float_poly_points = float_poly_data.GetPoints()

m, n = fixed_poly_data.GetNumberOfPoints(), float_poly_data.GetNumberOfPoints()

fixed_points = np.zeros((m, 3), dtype=np.float64)
float_points = np.zeros((n, 3), dtype=np.float64)

for i in range(m): fixed_points[i] = fixed_poly_points.GetPoint(i)
for i in range(n): float_points[i] = float_poly_points.GetPoint(i)
plt.style.use('ggplot')
fig = plt.figure(figsize=(5, 8))
labels_1 = [r'$\mathbf{a}_i^1$', r'$\mathbf{b}_i^1$', r'$\mathbf{a}_i^2$', r'$\mathbf{b}_i^2$']
labels_2 = [r'$\mathbf{a}_j^1$', r'$\mathbf{b}_j^1$', r'$\mathbf{a}_j^2$', r'$\mathbf{b}_j^2$']
title_1 = 'control points on fracture #2-a'
title_2 = 'control points on fracture #3-a'
plot(fig, fixed_points, 'teal', 211, title_1, labels_1, cluster=16)
plot(fig, float_points, 'darkgreen', 212, title_2, labels_2, cluster=16)

plt.show()
# fig.savefig('main_secondary_axis_vertical.png', dpi=500)
# plt.close(fig)