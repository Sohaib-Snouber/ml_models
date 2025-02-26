import open3d as o3d
import numpy as np
import pandas as pd

# Load point cloud with normals
pcd = o3d.io.read_point_cloud("/home/sohaib/zivid_robot_project/src/ml_models/data/red_cylinder_normalized/multi_red_cylinder11_0_normalized.ply")

# Convert to numpy arrays
points = np.asarray(pcd.points)
normals = np.asarray(pcd.normals)
colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(points)

# Compute curvature (simplified)
curvature = np.linalg.norm(normals, axis=1).reshape(-1, 1)

# Concatenate features
features = np.hstack((points, normals, colors, curvature))

# Convert to DataFrame
columns = ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b", "curvature"]
df = pd.DataFrame(features, columns=columns)

# Save as CSV for ML training
df.to_csv("/home/sohaib/zivid_robot_project/src/ml_models/data/training_data/red_cylinders11_0_features.csv", index=False)
print(f"✅ Features extracted and saved as 'features.csv'")

# Add label column (all points belong to a red cylinder)
df["label"] = "red_cylinder"

# Save labeled dataset
df.to_csv("/home/sohaib/zivid_robot_project/src/ml_models/data/training_data/red_cylinders11_0_labeled_features.csv", index=False)
print(f"✅ Data labeled and saved as 'labeled_features.csv'")