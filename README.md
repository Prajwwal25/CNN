# CNN

## Overview
This project implements a deep learning solution for classifying steel surface defects using a Convolutional Neural Network (CNN) architecture. The model is trained on the NEU Metal Surface Defects dataset and can identify various types of defects, including scratches, pits, rolled surfaces, and inclusions. This system is designed to enhance quality control processes in steel manufacturing environments.

## Features
* **CNN Architecture**: Robust defect classification capabilities
* **Advanced Pre-processing**: Image pre-processing with data augmentation for better generalization
* **Multi-class Classification**: Capable of identifying multiple defect types
* **High Accuracy**: Achieved 85% test accuracy on NEU Metal Surface Defects dataset

## Installation

### Hardware Requirements
* Minimum 8GB RAM
* GPU support recommended for faster training

### Software Dependencies
Install required packages using pip:
```bash
pip install tensorflow keras opencv-python matplotlib
```

Required libraries:
* TensorFlow
* Keras
* OpenCV-Python (image processing)
* Matplotlib (visualization)

## Dataset Structure
The NEU Metal Surface Defects dataset should be organized as follows:
```plaintext
steel_defect_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── labels/
│       ├── image1.txt
│       └── image2.txt
└── test/
    ├── images/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── labels/
        ├── image1.txt
        └── image2.txt
```

### Data Split Ratio
* Training: 80%
* Validation: 20%
* Test: 10-20% (optional)

## Model Architecture

### Network Structure
```plaintext
Input Layer (128x128x3)
   ↓
Conv2D + ReLU + MaxPool2D
   ↓
Conv2D + ReLU + MaxPool2D
   ↓
Conv2D + ReLU + MaxPool2D
   ↓
Flatten
   ↓
Dense (128 units) + ReLU
   ↓
Dense (8 units) + Softmax
```

### Layer Details
1. **Input Layer**
   * Size: (128, 128, 3) - RGB image

2. **Convolutional Layers**
   * Three sets of layers
   * 3x3 kernels
   * ReLU activation
   * MaxPooling2D (2x2) after each conv layer

3. **Dense Layers**
   * First dense: 128 neurons with ReLU
   * Output: 8 neurons with Softmax

## Training Process

1. **Data Preparation**
   * Organize dataset according to the structure above
   * Configure hyperparameters in the code

2. **Training**
   ```bash
   # Run the training notebook
   jupyter notebook CNN.ipynb
   ```

3. **Model Saving**
   * Trained model saves as `metal_surface_defects_model.h5`

## Model Performance

### Metrics
* Test Accuracy: 85%
* Evaluation performed on NEU Metal Surface Defects dataset

## Usage

### Making Predictions
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('metal_surface_defects_model.h5')

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Make prediction
def predict_defect(image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    return prediction
```
