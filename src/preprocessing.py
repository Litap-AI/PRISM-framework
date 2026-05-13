import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv("data/raw/products.csv")

print("\n✅ Dataset Loaded Successfully")
print(df.head())

# Features
X = df[
    [
        "performance",
        "relevance",
        "innovation",
        "scalability",
        "monetization"
    ]
]

# Labels
y = df["success_label"]

# Encode labels
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\n✅ Labels Encoded")
print(label_encoder.classes_)

# Normalize features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n✅ Features Scaled")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\n✅ Train-Test Split Complete")

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# Create processed directory if missing
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save processed arrays
joblib.dump(X_train, "data/processed/X_train.pkl")
joblib.dump(X_test, "data/processed/X_test.pkl")
joblib.dump(y_train, "data/processed/y_train.pkl")
joblib.dump(y_test, "data/processed/y_test.pkl")

# Save scaler + encoder
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\n✅ Preprocessing Complete")
print("Artifacts Saved Successfully")
