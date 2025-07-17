# model.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape images
train_images = np.expand_dims(train_images.astype('float32') / 255, -1)
test_images = np.expand_dims(test_images.astype('float32') / 255, -1)

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Create CNN model
def creation():
    model = models.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = creation()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=15, validation_split=0.3, verbose=1)

model.save("mnist_model.h5")
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)
