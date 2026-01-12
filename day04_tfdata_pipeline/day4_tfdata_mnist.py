import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# 1) Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)

# 2) Normalize (0-255 -> 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3) Create tf.data.Dataset pipeline
BATCH_SIZE = 64

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("✅ Dataset pipeline created successfully!")

# 4) Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5) Train
print("🔥 Training started...")
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

# 6) Evaluate
print("✅ Evaluating...")
test_loss, test_acc = model.evaluate(test_dataset)
print("Test Accuracy:", test_acc)

# 7) Predict one sample
sample = x_test[0:1]
prediction = model.predict(sample)
print("Predicted class:", tf.argmax(prediction, axis=1).numpy()[0])
print("Actual label:", y_test[0])
