import pickle

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get the mapping of labels to class indices
print("Class Mapping (Index → Color):")
for idx, class_name in enumerate(label_encoder.classes_):
    print(f"Class {idx}: {class_name}")
