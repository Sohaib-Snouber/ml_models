import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("labeled_colors.csv")

# Extract features (RGB) and labels (Color Name)
X = df[["R", "G", "B"]].values  # RGB values
Y = df["Color_Name"].values        # Color labels

# Normalize RGB values (scale to 0-1)
X = X / 255.0

# Split data into training (80%) and testing (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, Y_train)

# Evaluate the model
Y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

# Save the trained model
joblib.dump(clf, "color_classifier.pkl")
print("Model saved as 'color_classifier.pkl'")
