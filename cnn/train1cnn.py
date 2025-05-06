import os
import sys
import io

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = r'C:\Users\DELL\Desktop\minor\Plant-Disease-Detection-Main\train'
val_dir = r'C:\Users\DELL\Desktop\minor\Plant-Disease-Detection-Main\valid'

# Image size
image_size = (128, 128)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# CNN Model
model = Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on full dataset
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),   # All training data
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(val_generator)     # All validation data
)

# Save the model
model.save("plant_disease_cnn_model.h5")
print("Training complete and saved.")
