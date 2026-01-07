import tensorflow as tf
import numpy as np

# Dataset
# Feature: number of suspicious words
# Label: 0 = Not Spam, 1 = Spam
x = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x, y, epochs=200, verbose=0)

# Test data
test_data = np.array([[1], [2], [4], [5]], dtype=float)
predictions = model.predict(test_data)

print("Predictions:")
for value, prob in zip(test_data, predictions):
    result = "SPAM" if prob > 0.5 else "NOT SPAM"
    print(f"Input {value[0]} → {result} ({prob[0]:.2f})")
