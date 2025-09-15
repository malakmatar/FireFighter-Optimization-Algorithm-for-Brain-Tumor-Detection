"""
Baseline VGG16 Model

This script:
- Loads preprocessed MRI image data (train/val/test)
- Builds a VGG16-based CNN using transfer learning
- Trains the model and evaluates it on a held-out test set
- Uses callbacks (EarlyStopping, ModelCheckpoint)
- Plots accuracy/loss curves and saves training logs
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import json
import time


train_X = np.load("train_X.npy")
train_y = np.load("train_y.npy")
val_X = np.load("val_X.npy")
val_y = np.load("val_y.npy")
test_X = np.load("test_X.npy")
test_y = np.load("test_y.npy")

#normalization
train_X = train_X.astype('float32') / 255.0
val_X = val_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

#one hot encoding labels for the tumors (or no tumor)
num_classes = len(np.unique(train_y))
train_y = to_categorical(train_y, num_classes)
val_y = to_categorical(val_y, num_classes)
test_y = to_categorical(test_y, num_classes)


train_X, train_y = shuffle(train_X, train_y, random_state=42)


input_tensor = Input(shape=(224, 224, 3))
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

#freeze convolutional base
for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "vgg16_best_model.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

#training
start_time = time.time()

history = model.fit(
    train_X, train_y,
    validation_data=(val_X, val_y),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

training_duration = time.time() - start_time
print(f"Training completed in {training_duration:.2f} seconds")

#testing
test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")


with open("history.json", "w") as f:
    json.dump(history.history, f)


model.save("vgg16_baseline_model.h5")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
