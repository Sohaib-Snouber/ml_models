import torch
import torch.nn as nn
import pickle

# Load trained model architecture
class ColorClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load label encoder to determine number of classes
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    num_classes = len(label_encoder.classes_)  # Automatically detect class count

# Load trained model
model = ColorClassifier(num_classes)
model.load_state_dict(torch.load("color_classifier.pth"))
model.eval()  # Set to evaluation mode

# Define example input (matching your model's input shape)
example_input = torch.rand(1, 3)  # A batch size of 1 with 3 input features (RGB values)

# Export to ONNX
onnx_filename = "color_classifier.onnx"
torch.onnx.export(
    model,                      # PyTorch model
    example_input,              # Example input tensor
    onnx_filename,              # Output file name
    export_params=True,         # Store trained weights inside the model file
    opset_version=11,           # ONNX version
    input_names=["input"],      # Name of input layer
    output_names=["output"],    # Name of output layer
    dynamic_axes={              # Allow batch size to be dynamic
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

print(f"Model successfully converted to ONNX format: {onnx_filename}")
