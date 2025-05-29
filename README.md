# Face Mask Recognition 🚀😷

A deep learning model that detects whether a person is wearing a face mask in real-time or in static images, built with TensorFlow/Keras and OpenCV.

## Features ✨
- **Image-based prediction** for static images
- **High accuracy** (trained on 1000+ images)
- **Mobile-friendly** lightweight model architecture
- **Easy deployment** with pre-trained weights

## Tech Stack 💻
- Python 3.8+
- TensorFlow 2.x / Keras
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Installation ⚙️

### Prerequisites
```bash
git clone https://github.com/qoderepo-admin/face-mask-rekognition.git
cd face-mask-rekognition

**### Usage** 🛠️
1. Real-time Webcam Detection

python src/detect.py

2. Predict on Single Image

python src/predict.py

3. Train Model (Optional)

python src/train.py

## **Dataset 📊**
Trained on:

Data from https://github.com/prajnasb/observations/tree/master/experiements/data

## **Dataset structure:**

data/
├── with_mask/
├── without_mask/

## **Model Architecture 🧠**
python
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(2, activation="softmax"),
    ]
)
**Accuracy:** 96.2% on test set

## **Contributing 🤝**
Pull requests are welcome! For major changes, please open an issue first.


