import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(path):
    mesh = trimesh.load(path)
    vertices = mesh.vertices
    faces = mesh.faces
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        vertices[:, 0], vertices[:, 1], vertices[:, 2],
        triangles=faces, cmap='Spectral', lw=0.5, alpha=1.0
    )
    ax.set_axis_off()
    plt.show()