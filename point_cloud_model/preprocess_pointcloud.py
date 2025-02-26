import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder/multi_red_cylinder11_0.ply")

# Remove statistical outliers (noise)
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
pcd_cleaned = pcd.select_by_index(ind)

# Save cleaned point cloud
o3d.io.write_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder_cleaned/multi_red_cylinder11_0_cleaned.ply", pcd_cleaned)

# Print results
print(f"Original Points: {len(pcd.points)}")
print(f"Filtered Points: {len(pcd_cleaned.points)}")

# Visualize the cleaned point cloud
o3d.visualization.draw_geometries([pcd_cleaned])

# Normalize point cloud to maintain full detail
points = np.asarray(pcd_cleaned.points)
center = np.mean(points, axis=0)
points -= center
max_dist = np.max(np.linalg.norm(points, axis=1))
points /= max_dist  # Scale to [-1, 1]

# Recreate point cloud with normalized data
pcd_normalized = o3d.geometry.PointCloud()
pcd_normalized.points = o3d.utility.Vector3dVector(points)

# Preserve colors (if available)
if pcd_cleaned.has_colors():
    pcd_normalized.colors = pcd_cleaned.colors

# Save
o3d.io.write_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder_normalized/multi_red_cylinder11_0_normalized.ply", pcd_normalized)

print("Full-resolution point cloud successfully normalized!")
o3d.visualization.draw_geometries([pcd_normalized])

# Estimate surface normals
pcd_normalized.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Save with normals
o3d.io.write_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder_normalized/multi_red_cylinder11_0_normalized.ply", pcd_normalized)
print("✅ Normals computed and saved!")
