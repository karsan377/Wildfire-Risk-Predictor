import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# Parameters
img_size = (64, 64)  # You can increase this to 128x128 if your images are large
batch_size = 32
data_dir = 'fill in image directory'

# 1. Data generators (Keras will handle labels based on folder names)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# 2. Build the CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 3. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 5. Evaluate
val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator) > 0.5
print(classification_report(y_true, y_pred))

# 6. Save the model
model.save('wildfire_cnn_model.h5')
