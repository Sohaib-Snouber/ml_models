import open3d as o3d
import numpy as np
import pandas as pd
import colorsys

# Define step sizes
hue_steps = 361  # H from 0 to 360 (inclusive)
saturation_steps = np.arange(0.05, 1.05, 0.05)  # S from 0.05 to 1.0 in 0.05 steps
value_steps = np.arange(0.05, 1.05, 0.05)  # V from 0.05 to 1.0 in 0.05 steps

# Create data storage
points = []
colors = []
hue_labels = []  # Explicit hue labels

# Generate structured HSV dataset
for h in range(hue_steps):  # Loop through all hues (0-360 degrees)
    for s in saturation_steps:  # Loop through saturation values
        for v in value_steps:  # Loop through brightness values
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)

            # Assign structured positions: X increases with S, Y increases with V, Z is H
            x = s  # Saturation controls X position
            y = v  # Value (brightness) controls Y position
            z = h * 0.1  # Hue layer (separate each hue visually)

            points.append([x, y, z])
            colors.append([r, g, b])
            hue_labels.append(h)  # Store explicit hue label

# Convert to NumPy arrays
points = np.array(points)
colors = np.array(colors)
hue_labels = np.array(hue_labels).reshape(-1, 1)  # Convert to column format

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save as PLY file
o3d.io.write_point_cloud("labeled_structured_hsv.ply", pcd)

# Save as CSV file with explicit hue labels
df = pd.DataFrame(np.hstack((points, colors, hue_labels)), columns=["x", "y", "z", "r", "g", "b", "hue_label"])
df.to_csv("structured_hsv.csv", index=False)

print("✅ Structured HSV-based PLY file saved as 'structured_hsv.ply'")
print("✅ CSV dataset saved as 'structured_hsv.csv' with explicit hue labels.")
