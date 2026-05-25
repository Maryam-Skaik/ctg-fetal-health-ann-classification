# 🧠 Maternal Mortality (CTG) Multiclass Classification using Artificial Neural Networks

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge\&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge\&logo=scikitlearn)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📖 Project Overview

This project focuses on building and evaluating Artificial Neural Network (ANN) models for multiclass classification of fetal health conditions using Cardiotocography (CTG) data.

The notebook demonstrates a complete machine learning workflow including:

* Data preprocessing and cleaning
* Feature scaling
* Baseline neural network development
* Regularization using Dropout
* Hyperparameter optimization using KerasTuner
* Model evaluation and interpretation
* Confidence analysis using Softmax probabilities

The goal of the project is to classify fetal health status into:

* **Normal**
* **Suspect**
* **Pathological**

using neural networks implemented with TensorFlow/Keras.

---

## 🩹 Medical Context & Problem Statement

Cardiotocography (CTG) is a medical monitoring technique used during pregnancy to evaluate fetal well-being.

The dataset used in this project contains numerical measurements extracted from CTG exams, including patterns related to:

* Fetal heart rate
* Accelerations and decelerations
* Uterine contractions
* Variability measurements
* Abnormal fetal behavior indicators

### 🎯 Project Objective

The main objective is to build a deep learning model capable of predicting fetal health condition based on CTG measurements.

The target variable is:

```python
fetal_health
```

The model classifies fetal condition into three categories:

| Class        | Medical Meaning                                 |
| ------------ | ----------------------------------------------- |
| Normal       | Healthy fetal condition                         |
| Suspect      | Potential abnormalities requiring monitoring    |
| Pathological | High-risk condition requiring medical attention |

### 💡 Real-World Importance

Accurate fetal health prediction systems can support medical professionals by:

* Assisting early detection of fetal risk
* Reducing delayed diagnosis
* Supporting clinical decision-making
* Prioritizing high-risk pregnancy cases
* Improving maternal and fetal healthcare outcomes

This makes multiclass classification especially valuable in healthcare AI applications where prediction reliability is critical.

---

## 🩺 About the Dataset

The dataset contains CTG (Cardiotocography) measurements collected during pregnancy.

Cardiotocography is commonly used to monitor:

* Fetal heart rate
* Uterine contractions
* Overall fetal well-being

### 🎯 Target Variable

The original target column is:

```python
fetal_health
```

Original classes:

| Original Label | Meaning      |
| -------------- | ------------ |
| 1              | Normal       |
| 2              | Suspect      |
| 3              | Pathological |

For compatibility with TensorFlow multiclass classification requirements, the target was transformed into:

| Encoded Label | Meaning      |
| ------------- | ------------ |
| 0             | Normal       |
| 1             | Suspect      |
| 2             | Pathological |

Transformation:

```python
y = df['fetal_health'] - 1
```

---

## ⚙️ Technologies & Libraries

### 🐍 Core Libraries

* Python
* NumPy
* Pandas
* Matplotlib

### 🤖 Machine Learning & Deep Learning

* Scikit-learn
* TensorFlow
* Keras
* KerasTuner

---

## 🧹 Data Preprocessing

Several preprocessing steps were applied before training the neural networks.

### ✅ Data Cleaning

The following checks were performed:

* Verified missing values
* Verified feature data types
* Removed duplicate rows

### 📊 Feature Scaling

`StandardScaler` was applied to normalize the input features.

Why scaling is important for neural networks:

* Faster convergence
* Stable gradient updates
* Improved optimization performance
* Better overall model training

---

## 🧠 Model Development

Three ANN approaches were implemented and compared.

---

## 1️⃣ Baseline ANN Model

The first model was designed as a simple feedforward neural network to establish baseline performance.

### 🏗️ Architecture

* Dense layer → 64 neurons (ReLU)
* Dense layer → 32 neurons (ReLU)
* Output layer → 3 neurons (Softmax)

### ⚡ Compilation

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy

### 📉 Training Optimization

EarlyStopping was used to reduce unnecessary training and prevent overfitting.

---

## 2️⃣ Improved ANN Model (Regularization)

The second model introduced Dropout layers to improve generalization and reduce overfitting.

### 🛡️ Regularization Strategy

Dropout randomly disables neurons during training, which:

* Reduces overfitting
* Improves generalization
* Makes the model more robust

### 🏗️ Architecture

* Dense(64, ReLU)
* Dropout(0.3)
* Dense(32, ReLU)
* Dropout(0.2)
* Dense(3, Softmax)

---

## 3️⃣ Hyperparameter-Tuned ANN Model

KerasTuner with Hyperband optimization was used to automatically search for better hyperparameters.

### 🔍 Tuned Parameters

The tuner searched for:

* Number of neurons
* Dropout rate
* Optimizer type

### ⚙️ Optimizers Tested

* Adam
* RMSprop
* Nadam

This produced the best-performing model in the project.

---

## 📈 Model Evaluation

Multiple evaluation methods were used.

### ✅ Classification Report

The models were evaluated using:

* Precision
* Recall
* F1-score
* Accuracy

### 📊 Confusion Matrix

Confusion matrices were used to visualize prediction performance across all classes.

---

## 🔬 Softmax Confidence Analysis

The project also explored Softmax probability confidence.

Example:

```python
max_probs = np.max(pred, axis=1)
```

This represents the confidence score of the model for each prediction.

### 💡 Why Confidence Matters

In medical AI systems, confidence analysis is important because:

* Low-confidence predictions may require human review
* High-confidence predictions are generally more reliable
* Thresholds can improve prediction precision

Example strategy:

* High confidence → accept prediction
* Low confidence → send for expert verification

---

## 📊 Key Findings

### 🥇 Best Performing Model

The Hyperparameter-Tuned ANN achieved the best overall performance.

#### Why it performed best:

* Better hyperparameter selection
* Improved generalization
* More balanced classification performance
* Reduced overfitting

---

## 🧠 Project Insights

Several important deep learning concepts were demonstrated throughout the project:

### ✅ Neural networks benefit significantly from:

* Proper preprocessing
* Feature scaling
* Early stopping
* Dropout regularization
* Hyperparameter optimization

### ✅ Medical classification challenges

The project also highlighted:

* Difficulty in separating borderline classes
* Importance of confidence-based prediction
* The challenge of multiclass medical classification

---

## ⭐ Final Note

This project represents an applied introduction to neural networks using TensorFlow/Keras and demonstrates a full deep learning workflow from preprocessing to model optimization and evaluation.

It also reflects iterative experimentation and practical ANN development practices commonly used in real-world machine learning projects.
