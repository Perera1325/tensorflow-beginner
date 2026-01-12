import tensorflow as tf
import numpy as np
import os

# Training data (simple linear relation)
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train
model.fit(x, y, epochs=300, verbose=0)

# Save model
model_path = "saved_model/linear_model"
model.save(model_path)
print("Model saved at:", model_path)

# Load model
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully")

# Predict using loaded model
prediction = loaded_model.predict(np.array([6.0]))
print("Prediction for 6:", prediction)
