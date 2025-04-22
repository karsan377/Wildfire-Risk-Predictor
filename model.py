import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Parameters
img_size = (128, 128)  # MobileNetV2 expects at least 96x96
batch_size = 32
data_dir = 'fill in image directory'

# 1. Data generators
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

# 2. Load MobileNetV2 base
base_model = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base

# 3. Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=output)

# 4. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 🔍 Add this block right here
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss Curve Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Evaluate
val_generator.reset()
y_true = val_generator.classes
y_pred = model.predict(val_generator) > 0.5
print(classification_report(y_true, y_pred))

# 7. Save the model
model.save('wildfire_transfer_model.h5')
