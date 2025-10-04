# Oral-Cancer-Detector  

**Deep Learning-Based Oral Cancer Detection Using Smartphone Images**

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

![Keras](https://img.shields.io/badge/Keras-DeepLearning-red.svg)

![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green.svg)





---




## 📘 Overview

**Oral Cancer Detector** is a deep learning project designed to detect **oral cancer from smartphone-based images** using **Convolutional Neural Networks (CNNs)**.  

It enables **early, affordable, and mobile-friendly cancer screening**, especially in **rural or low-resource regions**.

The model classifies oral cavity images into **Benign** or **Malignant** categories and achieves an **accuracy of 84.38%** using a dual-branch CNN combining **RGB and HSV color spaces**.

---

## 🚀 Key Features

- 📱 Works with smartphone images  

- 🧠 Uses Convolutional Neural Networks (CNN)  

- 🎨 Multi-color space feature extraction (RGB + HSV)  

- 🧩 Achieves 84.38% accuracy with 0.84 F1-score  

- 🌍 Designed for rural and low-resource healthcare  

- ⚡ Lightweight and deployable on TensorFlow Lite  




---




## 🧩 Project Architecture
📂 oral-cancer-detector ├── dataset/                  # Training and test images ├── models/                   # Trained model files (.h5) ├── notebooks/                # Colab/Kaggle notebooks ├── src/ │   ├── data_preprocessing.py │   ├── train_model.py │   ├── evaluate_model.py │   └── utils.py ├── requirements.txt ├── README.md └── LICENSE

\## ⚙️ Tools & Technologies

\- \*\*Language:\*\* Python 3.10  

\- \*\*Frameworks:\*\* TensorFlow, Keras  

\- \*\*Libraries:\*\* NumPy, OpenCV, Scikit-learn, Matplotlib  

\- \*\*Environment:\*\* Google Colab / Kaggle  

\- \*\*Hardware Used:\*\* NVIDIA Tesla T4 GPU  

\---

\## 📊 Dataset

\- \*\*Source:\*\* Mendeley Data Repository  

\- \*\*Classes:\*\* Benign (165 images), Malignant (158 images)  

\- \*\*Image Size:\*\* 256×256 pixels  

\- \*\*Split:\*\* 80% Training | 20% Testing  

\- \*\*Augmentation:\*\* Flipping, rotation, zoom, and contrast  

\---

\## 🔬 Methodology

\### 1️⃣ Data Preprocessing

\- Normalize images (0–1 range)  

\- Apply color space transformations: RGB, HSV, YCrCb, Grayscale  

\- Perform data augmentation  

\### 2️⃣ Model Architecture

\- Dual-branch CNN for RGB and HSV  

\- Each branch extracts unique color features  

\- Fused features passed through dense layers  

\- Output: Binary classification (Benign / Malignant)  

\### 3️⃣ Training

\- \*\*Loss Function:\*\* Binary Cross Entropy  

\- \*\*Optimizer:\*\* Adam (lr = 0.001)  

\- \*\*Metrics:\*\* Accuracy, Precision, Recall, F1-score, ROC-AUC  

\- \*\*Early stopping\*\* to prevent overfitting  

\---

import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "CNN (Baseline)",
    "CNN + Augmentation",
    "CNN + Aug + Color (RGB+HSV)"
]

# Performance metrics
accuracy = [64.00, 71.88, 84.38]
precision = [0.65, 0.71, 0.84]
recall = [0.65, 0.70, 0.85]
f1_score = [0.64, 0.70, 0.84]

# Grouped bar chart setup
x = np.arange(len(models))
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - 1.5*width, accuracy, width, label="Accuracy")
bars2 = ax.bar(x - 0.5*width, precision, width, label="Precision")
bars3 = ax.bar(x + 0.5*width, recall, width, label="Recall")
bars4 = ax.bar(x + 1.5*width, f1_score, width, label="F1-Score")

# Labels & Formatting
ax.set_ylabel("Score")
ax.set_xlabel("Model Configuration")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.legend()

# Display values on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset
                    textcoords="offset points",
                    ha="center", va="bottom")

for bar_group in [bars1, bars2, bars3, bars4]:
    add_labels(bar_group)

plt.tight_layout()
plt.show()


🏆 \*\*Best Model:\*\* Dual-Branch CNN (RGB + HSV)

\---

\## 🖼️ Workflow Diagram

\`\`\`mermaid

graph TD;

    A\[Smartphone Image\] --&gt; B\[Preprocessing: Resize & Normalize\];

    B --&gt; C\[Color Space Conversion (RGB + HSV)\];

    C --&gt; D\[Dual CNN Branches\];

    D --&gt; E\[Feature Fusion\];

    E --&gt; F\[Dense Layers\];

    F --&gt; G\[Prediction: Benign or Malignant\];
