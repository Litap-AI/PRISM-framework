import joblib
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Load processed data
X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")

y_train = joblib.load("data/processed/y_train.pkl")
y_test = joblib.load("data/processed/y_test.pkl")

print("✅ Processed data loaded")

# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Build ANN model
model = Sequential()

model.add(Dense(
    64,
    activation='relu',
    input_shape=(5,)
))

model.add(Dropout(0.3))

model.add(Dense(
    32,
    activation='relu'
))

model.add(Dropout(0.2))

model.add(Dense(
    3,
    activation='softmax'
))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✅ ANN Model Built")

# Train model
history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=16,
    verbose=1
)

print("\n✅ Training Complete")

# Evaluate model
loss, accuracy = model.evaluate(
    X_test,
    y_test_cat
)

print(f"\nTest Accuracy: {accuracy:.4f}")

# Save model
os.makedirs("models", exist_ok=True)

model.save("models/prism_ann.keras")

print("\n✅ Model Saved Successfully")

# Plot Accuracy Graph
os.makedirs("results/graphs", exist_ok=True)

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("ANN Training Accuracy")

plt.legend()

plt.savefig(
    "results/graphs/training_accuracy.png"
)

print("\n✅ Accuracy Graph Saved")
