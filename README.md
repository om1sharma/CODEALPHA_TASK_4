# CODEALPHA_TASK_4
This project predicts diabetes, breast cancer, and heart disease using machine learning algorithms on medical datasets. It applies models like Logistic Regression and Random Forest to support early diagnosis and risk prediction based on patient health data.
🧬 Disease Prediction Model using Machine Learning
(Covers: Diabetes, Breast Cancer, and Heart Disease)

📌 Project Overview
This project focuses on building a robust machine learning-based system that can accurately predict the likelihood of three major diseases — Diabetes, Breast Cancer, and Heart Disease — using publicly available medical datasets. The aim is to assist in early detection and support preventive healthcare using data-driven predictions.

The model uses structured patient health data such as age, blood pressure, glucose levels, BMI, cholesterol, and other clinical features. It applies various classification algorithms to detect whether an individual is likely to suffer from any of the three diseases.

🩺 Diseases Covered
1. Diabetes Prediction
Dataset: Pima Indians Diabetes Dataset

Features: Glucose level, insulin, age, BMI, pregnancies, etc.

Goal: Predict if a patient is likely to develop diabetes.

2. Breast Cancer Detection
Dataset: Wisconsin Breast Cancer Dataset (WBCD)

Features: Cell radius, texture, perimeter, area, smoothness, etc.

Goal: Classify tumors as benign or malignant.

3. Heart Disease Prediction
Dataset: Cleveland Heart Disease Dataset

Features: Chest pain type, blood pressure, cholesterol, ECG, etc.

Goal: Predict the presence of heart disease based on symptoms.

🧠 Algorithms & Techniques Used
Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

XGBoost (Optional/Advanced)

Models are evaluated using metrics like:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC-AUC Curve

🔧 Workflow
Load datasets for each disease

Clean and preprocess the data (handling missing values, normalization)

Train multiple ML models

Compare model performances

Predict outcomes on test data or user input

Visualize results using confusion matrices and accuracy/loss graphs

📊 Key Features
Predicts disease risks using real medical data

Supports multi-model comparison

Clean, modular code

Can be extended to a GUI or Web App (like using Flask or Streamlit)

🌐 Real-World Use Cases
Early detection tools for healthcare providers

Health risk calculators on mobile/web platforms

Patient monitoring systems

Educational tools for students and researchers
