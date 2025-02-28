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

# Convert model to TorchScript
example_input = torch.rand(1, 3)  # Example input for tracing
traced_script_module = torch.jit.trace(model, example_input)

# Save TorchScript model
traced_script_module.save("color_classifier_scripted.pt")

print(f"Model converted and saved as color_classifier_scripted.pt with {num_classes} classes!")
