import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from train_color_classifier_pytorch import ColorClassifier  # Ensure this matches your model's definition

# Load dataset
df = pd.read_csv("labeled_colors.csv")
X = df[["R", "G", "B"]].values  # Use RGB as input
Y = df["Color_Name"].values

# Encode labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load trained model
model = ColorClassifier(num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("color_classifier.pth"))
model.eval()

# Evaluate accuracy
correct = 0
total = len(X)

with torch.no_grad():
    for i in range(total):
        input_tensor = torch.from_numpy(X[i]).float().unsqueeze(0)
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        if predicted_label == Y[i]:
            correct += 1

accuracy = correct / total
print(f"Model Accuracy: {accuracy:.2f}")
