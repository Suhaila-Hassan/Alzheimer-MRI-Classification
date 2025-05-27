# Alzheimer MRI Classification
Project leverages Convolutional Neural Networks (CNNs) using TensorFlow and Keras to classify MRI brain scans into four categories of Alzheimer's Disease:
- Mild Demented
- Moderate Demented
- Non Demented
- Very Mild Demented

# Project Workflow
### 1. Import Libraries
Libraries Used
- Data Manipulation & Visualization: numpy, pandas, matplotlib, seaborn, cv2, PIL
- ML/DL Frameworks: TensorFlow, Keras
- Model Evaluation: sklearn.metrics
- Imbalance Handling: imblearn.SMOTE
- System & OS: os, pathlib, sys
- Utilities: splitfolders, warnings

### 2. Dataset Acquisition & Analysis
- The dataset is uploaded and unzipped from Google Drive.
- Images are organized into 4 class folders.
- A bar chart visualizes the number of images per class.

### 3. Data Preprocessing
- Dataset is split into train, validation, and test using splitfolders.
- Image size is standardized to 128x128, and batch size is set to 32.
- The datasets are preprocessed using caching, shuffling, and prefetching to optimize training performance.
- Class imbalance is identified and addressed in Model 2 using SMOTE.

### 4. CNN Models
#### Model 1: Basic CNN
A sequential CNN model is defined with:
- Rescaling → Conv2D → MaxPooling → Dense → Softmax
- Trained for 20 epochs.
- Performance metrics and loss are visualized.

#### Model 2: CNN with Resampling
- Class imbalance is addressed using SMOTE.
- A new CNN model is trained on the resampled data.
- Same architecture as Model 1.

### Evaluation Metrics:
- Accuracy
- Confusion matrix
- Classification report
- Visualization of test images with actual and predicted label

### 5. Results
Resampling dataset did not improve classification accuracy for small classes.
