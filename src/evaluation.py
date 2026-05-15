import joblib
import os
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Load model
model = load_model("models/prism_ann.keras")

print("✅ Model Loaded")

# Load test data
X_test = joblib.load("data/processed/X_test.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

# Load label encoder
label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# Predict
y_pred_probs = model.predict(X_test)

# Convert probabilities → labels
y_pred = y_pred_probs.argmax(axis=1)

print("\n✅ Predictions Generated")

# Classification Report
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
)

print("\n===== Classification Report =====")
print(report)

# Save report
os.makedirs("results/reports", exist_ok=True)

with open(
    "results/reports/classification_report.txt",
    "w"
) as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(ax=ax)

plt.title("PRISM ANN Confusion Matrix")

# Save graph
os.makedirs(
    "results/graphs",
    exist_ok=True
)

plt.savefig(
    "results/graphs/confusion_matrix.png"
)

print("\n✅ Confusion Matrix Saved")
print("✅ Evaluation Complete")
