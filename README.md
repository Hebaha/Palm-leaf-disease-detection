# 🌴 Palm Leaf Disease Detection using Deep Learning

A Streamlit-based web application that detects palm leaf diseases using a deep learning model (EfficientNetB0).  
The application allows users to upload palm leaf images and predicts the disease class with confidence.

---
## 🛠️ Technologies Used

### 🔹 Programming Language
- **Python** – Core language used for model loading, preprocessing, and app logic

### 🔹 Deep Learning
- **TensorFlow** – Deep learning framework
- **Keras** – High-level API for model development and inference
- **EfficientNetB0** – Pretrained CNN architecture for image classification

### 🔹 Web Framework
- **Streamlit** – Used to build the interactive web application

### 🔹 Image Processing
- **Pillow (PIL)** – Image loading and preprocessing
- **NumPy** – Numerical operations on image arrays

### 🔹 Model & Data Handling
- **.keras Model Format** – Saved trained deep learning model
- **JSON** – Used for class label mapping (`class_labels.json`)

### 🔹 Deployment & Environment
- **Streamlit Cloud** – Cloud deployment platform
- **Linux-Compatible TensorFlow** – For cloud execution
- **Virtual Environment (venv)** – Dependency isolation

### 🔹 Version Control
- **Git** – Source code version control
- **GitHub** – Repository hosting and collaboration


## 🚀 Features

- Upload palm leaf images
- Deep learning–based disease classification
- EfficientNetB0 pretrained model
- User-friendly Streamlit interface
- Fast and accurate predictions
- Streamlit Cloud deployment ready

---

## 🧠 Model Used

- **Architecture:** EfficientNetB0  
- **Framework:** TensorFlow / Keras  
- **Model File:** `EfficientNetB0_palm_disease_model.keras`

---

## 📁 Project Structure

├── app.py
├── EfficientNetB0_palm_disease_model.keras
├── class_labels.json
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md

