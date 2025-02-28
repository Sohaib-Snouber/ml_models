# Color Classification with Machine Learning

This repository contains a machine learning model for **color classification**, trained on labeled colors using **PyTorch**. The model can classify colors based on RGB input and is trained with a dataset containing **human-labeled colors**.

## 📌 Features
- **Custom dataset generation**: HSV color dataset generation scripts.
- **Color labeling tool**: Manual labeling tool for color classification.
- **Color classification model**: PyTorch-based neural network.
- **Point cloud data processing**: Prepares point cloud data for training.

---

## 🛠️ Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/Sohaib-Snouber/ml_models.git
cd ml_models
```

### 2️⃣ **Create & Activate a Virtual Environment**
```bash
python3 -m venv ml
source ml/bin/activate  # For Linux/macOS
ml\Scripts\activate     # For Windows (if applicable)
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
_(Ensure `requirements.txt` exists, or install required libraries manually)_

### 4️⃣ **Run the Color Labeling Tool**
This tool allows you to manually label randomly generated colors.
```bash
python color_labeling_tool.py
```
- The tool will display a **random color**.
- You **enter the color name**, and the data gets saved in `labeled_colors.csv`.

---

## 📊 **Training the Color Classifier**
### **Train using PyTorch**
```bash
python train_color_classifier_pytorch.py
```
- This will train the model and save it as `color_classifier.pth`.

---

## 🏆 **Evaluating the Model**
```bash
python evaluate_color_classifier.py
```
- This will load `color_classifier.pth` and print the model's **accuracy**.

---

## 📡 **Using the Model for Prediction**
```python
import torch
import numpy as np
from train_color_classifier_pytorch import ColorClassifier

# Load trained model
model = ColorClassifier(num_classes=14)  # Adjust based on your dataset
model.load_state_dict(torch.load("color_classifier.pth"))
model.eval()

# Example RGB input
input_color = np.array([150, 169, 111])  # Example RGB
input_tensor = torch.tensor([input_color], dtype=torch.float32)

# Make prediction
output = model(input_tensor)
predicted_label = torch.argmax(output, dim=1).item()
print(f"Predicted Color Index: {predicted_label}")
```

---

## 🏗 **Point Cloud Data Processing**
This project also contains **point cloud feature extraction** for **object detection**.

### **Extract Features from a Point Cloud**
```bash
python point_cloud_model/extract_features.py
```
- Loads `.ply` files from `point_cloud_model/data/`.
- Extracts **color features** for training.

---

## 🔧 **Troubleshooting**
- If **OpenCV is missing**, install it using:
  ```bash
  pip install opencv-python
  ```
- If **PyTorch is missing**, install it with:
  ```bash
  pip install torch torchvision
  ```
- Ensure `labeled_colors.csv` exists before training the model.

---

## 📜 **License**
This project is open-source. Feel free to modify and improve it.

