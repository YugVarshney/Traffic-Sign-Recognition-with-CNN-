# 🛑 German Traffic Sign Classification (GTSRB)

This project builds a **deep learning model** using **TensorFlow** to classify German Traffic Signs from the [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).  
The model is trained on **43 traffic sign classes** and achieves high accuracy using a CNN (Convolutional Neural Network).

---

## 📂 Dataset
We use the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset available on Kaggle.  
To access the dataset:
1. Create a Kaggle account and generate your **API key (`kaggle.json`)** from [Kaggle Account Settings](https://www.kaggle.com/account).
2. Upload `kaggle.json` to your Colab/working directory.

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Download and unzip the dataset:


!kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
!unzip gtsrb-german-traffic-sign.zip -d dataset/

## ⚙️ Installation
Install dependencies before running the notebook:

!pip install tensorflow==2.19.1 pandas scikit-learn matplotlib kaggle
## 🏗️ Project Workflow
Load Dataset

Train images are loaded from dataset/Train/ with 43 class folders.

Test data is taken from dataset/Test.csv.

Preprocessing

Images resized to 30×30 pixels.

Normalization (/255.0).

Train-validation split (80-20).

Data Augmentation

Rotation, zoom, shift, shear, and fill-mode transformations.

## Model Architecture
A Convolutional Neural Network (CNN) with:

3 Convolution + MaxPooling layers

Flatten layer

Dense (128) + Dropout

Output layer (Softmax with 43 classes)

Training

Optimizer: Adam

Loss: sparse_categorical_crossentropy

Epochs: 15

Batch Size: 32

Evaluation

Metrics: Accuracy, Weighted F1-score

## 📊 Results
✅ Test Accuracy: 95.11%
✅ Weighted F1-score: 95.09%

The model performs robustly and generalizes well to unseen traffic sign images.

## 🚀 Usage
Training
history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(X_val, y_val)
)
### Save Model

model.save('traffic_sign_model.h5')

### Evaluate

from sklearn.metrics import accuracy_score, f1_score

y_pred = np.argmax(model.predict(X_test), axis=-1)
print("Test Accuracy:", accuracy_score(labels_test, y_pred))
print("Weighted F1-score:", f1_score(labels_test, y_pred, average='weighted'))

### Predict Single Image

from PIL import Image
import numpy as np

def predict_single(img_path):
    img = Image.open(img_path).resize((30,30))
    x = np.expand_dims(np.array(img)/255.0, axis=0)
    pred_class = np.argmax(model.predict(x), axis=-1)
    print("Predicted class:", class_labels[pred_class[0]])

predict_single("sample_image.png")

## 📌 Class Labels
The dataset contains 43 traffic sign categories, including:

Speed limits (20–120 km/h)

No passing zones

Stop, Yield, Priority Road

Caution signs (Slippery road, Pedestrians, Children, Animals, etc.)

Directional signs (Turn left/right, Roundabout, Ahead only, etc.)

📁 Repository Structure

Copy code

├── Traffic_Sign_Recognition_using_CNN .ipynb          # Training + Evaluation Notebook

├── README.md               # Project Documentation

## 🛠️ Tech Stack
Python 3.9+

TensorFlow 2.19.1

Scikit-learn

Matplotlib

Pandas

Kaggle API

## 🙌 Acknowledgements
Dataset: Kaggle - GTSRB German Traffic Sign

Research inspiration: GTSRB Benchmark Dataset
