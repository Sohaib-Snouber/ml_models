import open3d as o3d
import numpy as np
import pandas as pd
import colorsys
import matplotlib.pyplot as plt

# Define the hue, saturation, and value steps
hue_steps = 360  # Full hue rotation (0-360°)
saturation_steps = np.arange(0.01, 1.01, 0.01)  # Saturation from 0.01 to 1.0
value_steps = np.arange(0.01, 1.01, 0.01)  # Brightness from 0.01 to 1.0

points = []
colors = []

# Create a structured grid
grid_size = int(np.ceil(np.sqrt(hue_steps * len(saturation_steps) * len(value_steps))))
spacing = 0.05  # Space between points for visualization

index = 0  # Counter for structured grid positioning
for hue in range(hue_steps):  # Loop through all hue values (0-360)
    for sat in saturation_steps:  # Loop through all saturation levels
        for val in value_steps:  # Loop through all brightness levels
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue / 360, sat, val)

            # Assign a structured position for visualization
            x = (index % grid_size) * spacing
            y = (index // grid_size) * spacing
            z = 0  # Keep all points on a plane

            points.append([x, y, z])
            colors.append([r, g, b])
            index += 1  # Move to the next grid position

# Convert to NumPy arrays
points = np.array(points)
colors = np.array(colors)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Save as PLY file
o3d.io.write_point_cloud("full_hsv_pointcloud.ply", pcd)

# Save as CSV for easy labeling
df = pd.DataFrame(np.hstack((points, colors)), columns=["x", "y", "z", "r", "g", "b"])
df.to_csv("full_hsv_dataset.csv", index=False)

print("✅ Full HSV-based point cloud saved as 'full_hsv_pointcloud.ply'")
print("✅ Full HSV dataset saved as 'full_hsv_dataset.csv' for manual labeling.")

# --- Visualization for Easy Labeling ---
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xticks([])  # Remove axis ticks for cleaner display
ax.set_yticks([])
ax.set_title("Full HSV-Based Structured Visualization")

# Convert RGB colors to hex for plotting
hex_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors[:grid_size**2]]

# Reshape into a 2D image for visualization
grid_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
for i in range(min(len(hex_colors), grid_size**2)):
    row, col = divmod(i, grid_size)
    grid_image[row, col] = hex_colors[i]

# Show structured color grid
ax.imshow(grid_image)
plt.show()
