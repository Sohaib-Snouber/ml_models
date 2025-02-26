import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Check PyTorch installation
print("✅ PyTorch Version:", torch.__version__)
print("✅ CUDA Available:", torch.cuda.is_available())

# Check Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
o3d.visualization.draw_geometries([pcd])

# Check Matplotlib
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Matplotlib Test")
plt.show()
