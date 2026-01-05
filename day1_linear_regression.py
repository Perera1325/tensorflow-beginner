import tensorflow as tf
import numpy as np

# Simple dataset
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile model
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train model
model.fit(x, y, epochs=500, verbose=0)

# Test prediction (FIXED)
prediction = model.predict(np.array([6.0]))
print("Prediction for 6:", prediction)
