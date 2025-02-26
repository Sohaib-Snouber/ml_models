import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder/multi_red_cylinder11_0.ply")

# Convert to numpy array
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) if pcd.has_colors() else None

# Print point cloud details
print(f"Loaded Point Cloud: {points.shape[0]} points")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
