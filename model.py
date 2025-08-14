import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Parameters
img_size = (128, 128)  # MobileNetV2 expects at least 96x96
batch_size = 32
data_dir = '/Users/karthik/Desktop/Projects/Wildfire/model'

# 1. Data generators (with validation split)
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

# 2. Load MobileNetV2 base (pretrained on ImageNet)
base_model = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base for transfer learning

# 3. Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification output

model = Model(inputs=base_model.input, outputs=output)

# 4. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# 6. Plot training and validation loss curves
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

# 7. Evaluate on validation data
val_generator.reset()
y_true = val_generator.classes

# Get probability predictions (for AUC-ROC)
y_pred_proba = model.predict(val_generator).ravel()  # Flatten to shape (num_samples,)

# Convert probabilities to binary predictions (for classification report)
y_pred = (y_pred_proba > 0.5).astype(int)

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Print AUC-ROC score
auc = roc_auc_score(y_true, y_pred_proba)
print(f"AUC-ROC: {auc:.4f}")

# 7.5 Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()


# 8. Save the trained model
model.save('wildfire_transfer_model.h5')
print("\nModel saved as 'wildfire_transfer_model.h5'")
