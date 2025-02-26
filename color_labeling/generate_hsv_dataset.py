import open3d as o3d
import numpy as np
import pandas as pd
import colorsys

# Define smaller step sizes
hue_steps = 360  # H from 0 to 359
saturation_steps = np.linspace(0.1, 1.0, 10)  # 10 Saturation levels from 0.1 to 1.0
value_steps = np.linspace(0.1, 1.0, 10)  # 10 Brightness levels from 0.1 to 1.0

# Create data storage
points = []
colors = []
hue_labels = []  # Explicit hue labels

# Improved Structured HSV Dataset with Clearer Separation
row_spacing = 0.2  # Increase spacing between hue rows for easier differentiation
column_spacing = 0.2  # Increase spacing between saturation-value pairs

for h in range(hue_steps):  # Loop through 360 hues (Each row = 1 hue)
    row_offset = h * row_spacing  # Separate hues in rows

    for i, s in enumerate(saturation_steps):  # Loop through 10 saturation values
        for j, v in enumerate(value_steps):  # Loop through 10 brightness values
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)

            # Assign structured positions with clear separation
            x = i * column_spacing  # Spread saturation steps along X-axis
            y = j * column_spacing  # Spread brightness steps along Y-axis
            z = row_offset  # Each hue gets its own row (clear separation)

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
o3d.io.write_point_cloud("small_hsv.ply", pcd)

# Save as CSV file with explicit hue labels
df = pd.DataFrame(np.hstack((points, colors, hue_labels)), columns=["x", "y", "z", "r", "g", "b", "hue_label"])
df.to_csv("small_hsv.csv", index=False)

print("✅ Smaller HSV dataset saved as 'small_hsv.ply' (for visualization).")
print("✅ CSV dataset saved as 'small_hsv.csv' (for easy labeling).")


o3d.visualization.draw_geometries([pcd])
