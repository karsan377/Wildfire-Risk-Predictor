import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# ------------------- Constants -------------------
MODEL_PATH = 'wildfire_transfer_model.h5'
IMG_SIZE = (128, 128)
SAMPLE_IMAGE_PATH = 'model/2high Risk/UCSD forest.jpg'
ALPHA = 0.5  # transparency for overlay

# ------------------- Load Model -------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------- Find Last Conv Layer -------------------
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name is None:
    raise ValueError("No Conv2D layer found in the model.")
else:
    print(f"Using last conv layer: {last_conv_layer_name}")

# ------------------- Grad-CAM Function -------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input],  # use model.input instead of raw tensor
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap) + 1e-8  # avoid divide by zero
    return heatmap.numpy()

# ------------------- Load and Preprocess Image -------------------
img = tf.keras.preprocessing.image.load_img(SAMPLE_IMAGE_PATH, target_size=IMG_SIZE)
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
img_uint8 = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)

# ------------------- Generate Heatmap -------------------
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# ------------------- Overlay Heatmap -------------------
heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

# Ensure both images are uint8 and same shape
superimposed_img = cv2.addWeighted(img_uint8, 1, heatmap_color, ALPHA, 0)

# ------------------- Display Result -------------------
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Grad-CAM Visualization")
plt.show()
