#!/usr/bin/env python
# coding: utf-8

# # CODEALPHA TASK-4 DISEASE PREDICTION FROM MEDICAL DATA

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                            precision_score, recall_score, f1_score, roc_auc_score, 
                            roc_curve, precision_recall_curve)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ======================================================================
# STEP 1: DATA LOADING AND PREPROCESSING FOR ALL DATASETS
# ======================================================================

print("\n" + "="*100)
print("🔹 STEP 1: LOADING AND PREPROCESSING MULTIPLE DISEASE DATASETS")
print("="*100 + "\n")

# Load datasets with proper column names
print("📂 Loading datasets from UCI ML Repository...")

# 1. Heart Disease Dataset
print("\n❤️ Heart Disease Dataset:")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
try:
    heart_df = pd.read_csv(heart_url, names=heart_cols)
    heart_df['target'] = heart_df['target'].apply(lambda x: 1 if x > 0 else 0)
    print(f"✔ Loaded {heart_df.shape[0]} records with {heart_df.shape[1]} features")
    
    # Handle missing values (marked as ? in this dataset)
    heart_df = heart_df.replace('?', np.nan)
    heart_df = heart_df.dropna()
    heart_df = heart_df.apply(pd.to_numeric)
except Exception as e:
    print(f"❌ Error loading heart dataset: {str(e)}")

# 2. Diabetes Dataset
print("\n🩸 Diabetes Dataset:")
diabetes_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"
try:
    diabetes_df = pd.read_csv(diabetes_url)
    print(f"✔ Loaded {diabetes_df.shape[0]} records with {diabetes_df.shape[1]} features")
    
    # Convert categorical features to numerical
    le = LabelEncoder()
    for col in diabetes_df.columns:
        if diabetes_df[col].dtype == 'object':
            diabetes_df[col] = le.fit_transform(diabetes_df[col])
except Exception as e:
    print(f"❌ Error loading diabetes dataset: {str(e)}")

# 3. Breast Cancer Dataset
print("\n🎗️ Breast Cancer Dataset:")
cancer_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
cancer_cols = ['id', 'clump_thickness', 'cell_size', 'cell_shape', 'marginal_adhesion',
               'single_epithelial', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli',
               'mitoses', 'class']
try:
    cancer_df = pd.read_csv(cancer_url, names=cancer_cols)
    print(f"✔ Loaded {cancer_df.shape[0]} records with {cancer_df.shape[1]} features")
    
    # Handle missing values and convert to binary classification
    cancer_df = cancer_df.replace('?', np.nan)
    cancer_df = cancer_df.dropna()
    cancer_df = cancer_df.drop('id', axis=1)
    cancer_df['class'] = cancer_df['class'].apply(lambda x: 1 if x == 4 else 0)
    cancer_df = cancer_df.apply(pd.to_numeric)
except Exception as e:
    print(f"❌ Error loading breast cancer dataset: {str(e)}")

# ======================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS FOR ALL DATASETS
# ======================================================================

print("\n" + "="*100)
print("🔹 STEP 2: EXPLORATORY DATA ANALYSIS FOR ALL DATASETS")
print("="*100 + "\n")

def plot_distribution(df, target_col, title):
    """Helper function to plot target distribution"""
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=target_col, data=df, palette=['#ff9999','#66b3ff'])
    plt.title(f'{title} Distribution', fontsize=14)
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        x = p.get_x() + p.get_width()/2
        y = p.get_height() + 10
        ax.annotate(percentage, (x, y), ha='center')
    plt.show()

# 1. Heart Disease EDA
print("\n❤️ Heart Disease Dataset Analysis:")
print("First 5 records:")
print(heart_df.head())
plot_distribution(heart_df, 'target', 'Heart Disease')

# 2. Diabetes EDA
print("\n🩸 Diabetes Dataset Analysis:")
print("First 5 records:")
print(diabetes_df.head())
plot_distribution(diabetes_df, 'class', 'Diabetes')

# 3. Breast Cancer EDA
print("\n🎗️ Breast Cancer Dataset Analysis:")
print("First 5 records:")
print(cancer_df.head())
plot_distribution(cancer_df, 'class', 'Breast Cancer')

# ======================================================================
# STEP 3: MODEL TRAINING FOR ALL DATASETS
# ======================================================================

print("\n" + "="*100)
print("🔹 STEP 3: TRAINING PREDICTION MODELS FOR ALL DISEASES")
print("="*100 + "\n")

# Define common models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

def train_and_evaluate(X, y, disease_name):
    """Train and evaluate models for a given dataset"""
    print(f"\n💊 Training {disease_name} Prediction Models...")
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize metrics DataFrame
    metrics_list = []
    results = {}
    
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probas)
        
        # Store results
        results[name] = {'preds': preds, 'probas': probas}
        
        # Append metrics to list
        metrics_list.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC AUC': roc_auc
        })
        
        # Print metrics
        print(f"✔ {name} Performance:")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F1 Score:  {f1*100:.2f}%")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy', 'Disease'],
                    yticklabels=['Healthy', 'Disease'])
        plt.title(f'{name} - {disease_name}\nConfusion Matrix', fontsize=12)
        plt.xlabel('Predicted', fontsize=10)
        plt.ylabel('Actual', fontsize=10)
        plt.show()
    
    # Convert metrics list to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Ensemble prediction
    print("\n🤝 Creating ensemble model...")
    preds_matrix = np.vstack([results[m]['preds'] for m in results]).T
    ensemble_preds = mode(preds_matrix, axis=1).mode.flatten()
    
    # Calculate ensemble metrics
    acc_ensemble = accuracy_score(y_test, ensemble_preds)
    prec_ensemble = precision_score(y_test, ensemble_preds)
    rec_ensemble = recall_score(y_test, ensemble_preds)
    f1_ensemble = f1_score(y_test, ensemble_preds)
    
    print("\n✔ Ensemble Performance:")
    print(f"  Accuracy:  {acc_ensemble*100:.2f}%")
    print(f"  Precision: {prec_ensemble*100:.2f}%")
    print(f"  Recall:    {rec_ensemble*100:.2f}%")
    print(f"  F1 Score:  {f1_ensemble*100:.2f}%")
    
    # Plot confusion matrix for ensemble
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, ensemble_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'])
    plt.title(f'Ensemble - {disease_name}\nConfusion Matrix', fontsize=12)
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('Actual', fontsize=10)
    plt.show()
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'metrics': metrics_df.sort_values(by='Accuracy', ascending=False)
    }

# Train models for each disease
heart_results = train_and_evaluate(
    heart_df.drop('target', axis=1), 
    heart_df['target'], 
    "Heart Disease"
)

diabetes_results = train_and_evaluate(
    diabetes_df.drop('class', axis=1), 
    diabetes_df['class'], 
    "Diabetes"
)

cancer_results = train_and_evaluate(
    cancer_df.drop('class', axis=1), 
    cancer_df['class'], 
    "Breast Cancer"
)

# ======================================================================
# STEP 4: PREDICTION FUNCTION FOR ALL DISEASES
# ======================================================================

print("\n" + "="*100)
print("🔹 STEP 4: COMPREHENSIVE PREDICTION FUNCTION")
print("="*100 + "\n")

# ======================================================================
# STEP 4: PREDICTION FUNCTION FOR ALL DISEASES (UPDATED)
# ======================================================================

def predict_health_status(patient_data):
    """Predict all three diseases for a given patient"""
    print("\n" + "🔍"*50)
    print("🔍 HEALTH STATUS PREDICTION FOR PATIENT:")
    print(f"  Age: {patient_data['age']}")
    print(f"  Sex: {'Male' if patient_data['sex'] == 1 else 'Female'}")
    print(f"  Blood Pressure: {patient_data.get('trestbps', 'N/A')}")
    print(f"  Cholesterol: {patient_data.get('chol', 'N/A')}")
    print(f"  Glucose Level: {patient_data.get('glucose', 'N/A')}")
    
    results = {}
    
    # 1. Heart Disease Prediction
    print("\n❤️ Heart Disease Prediction:")
    heart_features = heart_results['feature_names']
    
    # Create input array with only the features the model was trained on
    heart_input = []
    for f in heart_features:
        if f in patient_data:
            heart_input.append(patient_data[f])
        else:
            # If feature is missing, use median value from training data
            median_val = heart_df[f].median()
            heart_input.append(median_val)
            print(f"  ⚠️ Using median value ({median_val:.2f}) for missing feature: {f}")
    
    heart_input = np.array([heart_input])
    heart_input = heart_results['scaler'].transform(heart_input)
    
    heart_preds = {}
    for name, model in heart_results['models'].items():
        try:
            pred = model.predict(heart_input)[0]
            proba = model.predict_proba(heart_input)[0][1]
            heart_preds[name] = {'pred': pred, 'confidence': proba}
        except Exception as e:
            print(f"  ❗ Error in {name}: {str(e)}")
            heart_preds[name] = {'pred': 0, 'confidence': 0.5}
    
    if heart_preds:
        # Get the mode (most common prediction) from all models
        pred_values = [v['pred'] for v in heart_preds.values()]
        try:
            heart_ensemble = mode(pred_values).mode[0]
        except:
            # If mode fails (all predictions equal), use majority
            heart_ensemble = int(np.mean(pred_values) > 0.5)
        heart_confidence = np.mean([v['confidence'] for v in heart_preds.values()])
        
        print("  Individual Model Predictions:")
        for name, pred in heart_preds.items():
            print(f"    {name:<20}: {'Disease' if pred['pred'] else 'Healthy'} "
                  f"(confidence: {pred['confidence']*100:.1f}%)")
        print(f"  🎯 Ensemble Prediction: {'Heart Disease Risk' if heart_ensemble else 'Healthy Heart'} "
              f"(confidence: {heart_confidence*100:.1f}%)")
    else:
        print("  ❗ No valid heart disease predictions could be made")
        heart_ensemble = 0
        heart_confidence = 0.5
    
    # 2. Diabetes Prediction
    print("\n🩸 Diabetes Prediction:")
    diabetes_features = diabetes_results['feature_names']
    
    diabetes_input = []
    for f in diabetes_features:
        if f in patient_data:
            diabetes_input.append(patient_data[f])
        else:
            # If feature is missing, use median value from training data
            median_val = diabetes_df[f].median()
            diabetes_input.append(median_val)
            print(f"  ⚠️ Using median value ({median_val:.2f}) for missing feature: {f}")
    
    diabetes_input = np.array([diabetes_input])
    diabetes_input = diabetes_results['scaler'].transform(diabetes_input)
    
    diabetes_preds = {}
    for name, model in diabetes_results['models'].items():
        try:
            pred = model.predict(diabetes_input)[0]
            proba = model.predict_proba(diabetes_input)[0][1]
            diabetes_preds[name] = {'pred': pred, 'confidence': proba}
        except Exception as e:
            print(f"  ❗ Error in {name}: {str(e)}")
            diabetes_preds[name] = {'pred': 0, 'confidence': 0.5}
    
    if diabetes_preds:
        pred_values = [v['pred'] for v in diabetes_preds.values()]
        try:
            diabetes_ensemble = mode(pred_values).mode[0]
        except:
            diabetes_ensemble = int(np.mean(pred_values) > 0.5)
        diabetes_confidence = np.mean([v['confidence'] for v in diabetes_preds.values()])
        
        print("  Individual Model Predictions:")
        for name, pred in diabetes_preds.items():
            print(f"    {name:<20}: {'Diabetes Risk' if pred['pred'] else 'Healthy'} "
                  f"(confidence: {pred['confidence']*100:.1f}%)")
        print(f"  🎯 Ensemble Prediction: {'Diabetes Risk' if diabetes_ensemble else 'Healthy'} "
              f"(confidence: {diabetes_confidence*100:.1f}%)")
    else:
        print("  ❗ No valid diabetes predictions could be made")
        diabetes_ensemble = 0
        diabetes_confidence = 0.5
    
    # 3. Breast Cancer Prediction (for females only)
    print("\n🎗️ Breast Cancer Prediction:")
    if patient_data['sex'] == 0:  # Female
        cancer_features = cancer_results['feature_names']
        
        cancer_input = []
        for f in cancer_features:
            if f in patient_data:
                cancer_input.append(patient_data[f])
            else:
                # If feature is missing, use median value from training data
                median_val = cancer_df[f].median()
                cancer_input.append(median_val)
                print(f"  ⚠️ Using median value ({median_val:.2f}) for missing feature: {f}")
        
        cancer_input = np.array([cancer_input])
        cancer_input = cancer_results['scaler'].transform(cancer_input)
        
        cancer_preds = {}
        for name, model in cancer_results['models'].items():
            try:
                pred = model.predict(cancer_input)[0]
                proba = model.predict_proba(cancer_input)[0][1]
                cancer_preds[name] = {'pred': pred, 'confidence': proba}
            except Exception as e:
                print(f"  ❗ Error in {name}: {str(e)}")
                cancer_preds[name] = {'pred': 0, 'confidence': 0.5}
        
        if cancer_preds:
            pred_values = [v['pred'] for v in cancer_preds.values()]
            try:
                cancer_ensemble = mode(pred_values).mode[0]
            except:
                cancer_ensemble = int(np.mean(pred_values) > 0.5)
            cancer_confidence = np.mean([v['confidence'] for v in cancer_preds.values()])
            
            print("  Individual Model Predictions:")
            for name, pred in cancer_preds.items():
                print(f"    {name:<20}: {'Cancer Risk' if pred['pred'] else 'Healthy'} "
                      f"(confidence: {pred['confidence']*100:.1f}%)")
            print(f"  🎯 Ensemble Prediction: {'Breast Cancer Risk' if cancer_ensemble else 'Healthy'} "
                  f"(confidence: {cancer_confidence*100:.1f}%)")
        else:
            print("  ❗ No valid breast cancer predictions could be made")
            cancer_ensemble = 0
            cancer_confidence = 0.5
    else:
        print("  Not applicable for male patients")
        cancer_ensemble = 0
        cancer_confidence = 0.0
    
    return {
        'heart_disease': {
            'prediction': bool(heart_ensemble),
            'confidence': float(heart_confidence)
        },
        'diabetes': {
            'prediction': bool(diabetes_ensemble),
            'confidence': float(diabetes_confidence)
        },
        'breast_cancer': {
            'prediction': bool(cancer_ensemble) if patient_data['sex'] == 0 else None,
            'confidence': float(cancer_confidence) if patient_data['sex'] == 0 else None
        }
    }

# ======================================================================
# STEP 5: DEMONSTRATION WITH EXAMPLE PATIENTS
# ======================================================================

print("\n" + "="*100)
print("🔹 STEP 5: DEMONSTRATION WITH EXAMPLE PATIENTS")
print("="*100 + "\n")

# Example 1: Healthy Young Female
print("\n👩 Example 1: Healthy 28-year-old Female")
healthy_female = {
    'age': 28,
    'sex': 0,  # Female
    'trestbps': 110,  # Blood pressure
    'chol': 170,  # Cholesterol
    'glucose': 85,  # Glucose level
    'cp': 0,  # Chest pain type
    'fbs': 0,  # Fasting blood sugar
    'restecg': 0,  # ECG results
    'thalach': 175,  # Max heart rate
    'exang': 0,  # Exercise induced angina
    'oldpeak': 0.5,  # ST depression
    'slope': 1,  # Slope of peak exercise
    'ca': 0,  # Number of major vessels
    'thal': 2,  # Thalassemia
    'Polyuria': 0,  # Diabetes symptoms
    'Polydipsia': 0,
    'sudden weight loss': 0,
    'weakness': 0,
    'Polyphagia': 0,
    'Genital thrush': 0,
    'visual blurring': 0,
    'Itching': 0,
    'Irritability': 0,
    'delayed healing': 0,
    'partial paresis': 0,
    'muscle stiffness': 0,
    'Alopecia': 0,
    'Obesity': 0,
    'clump_thickness': 2,  # Cancer features
    'cell_size': 1,
    'cell_shape': 1,
    'marginal_adhesion': 1,
    'single_epithelial': 2,
    'bare_nuclei': 1,
    'bland_chromatin': 3,
    'normal_nucleoli': 1,
    'mitoses': 1
}
predict_health_status(healthy_female)

# Example 2: Middle-aged Male with Heart Disease Risk
print("\n👨 Example 2: 55-year-old Male with Heart Disease Risk")
heart_risk_male = {
    'age': 55,
    'sex': 1,  # Male
    'trestbps': 150,  # High blood pressure
    'chol': 280,  # High cholesterol
    'glucose': 95,  # Normal glucose
    'cp': 3,  # Chest pain type
    'fbs': 0,  # Fasting blood sugar
    'restecg': 1,  # ECG results
    'thalach': 140,  # Max heart rate
    'exang': 1,  # Exercise induced angina
    'oldpeak': 2.5,  # ST depression
    'slope': 2,  # Slope of peak exercise
    'ca': 1,  # Number of major vessels
    'thal': 3,  # Thalassemia
    'Polyuria': 0,  # Diabetes symptoms
    'Polydipsia': 0,
    'sudden weight loss': 0,
    'weakness': 0,
    'Polyphagia': 0,
    'Genital thrush': 0,
    'visual blurring': 0,
    'Itching': 0,
    'Irritability': 0,
    'delayed healing': 0,
    'partial paresis': 0,
    'muscle stiffness': 0,
    'Alopecia': 0,
    'Obesity': 0
}
predict_health_status(heart_risk_male)

# Example 3: Elderly Female with Multiple Risks
print("\n👵 Example 3: 65-year-old Female with Multiple Risks")
elderly_female = {
    'age': 65,
    'sex': 0,  # Female
    'trestbps': 145,  # High blood pressure
    'chol': 240,  # High cholesterol
    'glucose': 140,  # High glucose
    'cp': 2,  # Chest pain
    'fbs': 1,  # High fasting blood sugar
    'restecg': 0,
    'thalach': 130,
    'exang': 0,
    'oldpeak': 1.5,
    'slope': 1,
    'ca': 0,
    'thal': 2,
    'Polyuria': 1,  # Diabetes symptoms
    'Polydipsia': 1,
    'sudden weight loss': 1,
    'weakness': 1,
    'Polyphagia': 1,
    'Genital thrush': 0,
    'visual blurring': 1,
    'Itching': 0,
    'Irritability': 0,
    'delayed healing': 1,
    'partial paresis': 0,
    'muscle stiffness': 0,
    'Alopecia': 0,
    'Obesity': 1,
    'clump_thickness': 6,  # Cancer features
    'cell_size': 5,
    'cell_shape': 5,
    'marginal_adhesion': 4,
    'single_epithelial': 5,
    'bare_nuclei': 4,
    'bland_chromatin': 6,
    'normal_nucleoli': 4,
    'mitoses': 2
}
predict_health_status(elderly_female)

print("\n" + "="*100)
print("✅ COMPREHENSIVE DISEASE PREDICTION SYSTEM COMPLETE")
print("="*100 + "\n")


# In[ ]:




