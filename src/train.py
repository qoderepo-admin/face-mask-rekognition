import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path setup
data_dir = "../data"
categories = ["with_mask", "without_mask"]

# Step 1: Load images and labels
data = []
labels = []

for category in categories:
    # Get the directory of the current script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "..", "data", category)
    print(f"path={path}")
    class_num = categories.index(category)  # 0 or 1
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))  # Resize all to 100x100
            data.append(img)
            labels.append(class_num)
        except Exception as e:
            print("Error loading image:", img_name)

# Step 2: Prepare data
data = np.array(data) / 255.0  # Normalize pixels (0-1)
labels = to_categorical(labels, 2)

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Step 4: Create CNN model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax"),  # 2 classes
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 5: Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Step 6: Save the model
model.save(os.path.join(BASE_DIR, "..", "model/mask_detector.h5"))
print("Model saved to model/mask_detector.h5")
