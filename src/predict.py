import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model/mask_detector.h5")
model = load_model(model_path)

# Load image to test
image_to_predict = "sample2.jpg"
image_path = os.path.join(BASE_DIR, "..", "test_images", image_to_predict)
image = cv2.imread(image_path)
resized = cv2.resize(image, (100, 100)) / 255.0  # Normalize

# Make prediction
prediction = model.predict(np.expand_dims(resized, axis=0))[0]
print(f"prediction = {prediction}")
class_idx = np.argmax(prediction)
accuracy = np.max(prediction) * 100

# Print result
print("Prediction:", "With Mask" if class_idx == 0 else "Without Mask")
print(f"Accuracy: {accuracy}")
