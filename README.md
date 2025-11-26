# Heart Disease Prediction App ❤️

A machine learning-based web application to predict the risk of heart disease using key patient health metrics. The project is built with **Python, scikit-learn, and Streamlit** and allows users to enter important factors to estimate heart disease risk.

---

## Project Objective

The goal of this project is to build a **binary classification model** that predicts whether a patient has heart disease based on their health data, such as age, chest pain type, maximum heart rate, ST depression, and exercise-induced angina.

---

## Dataset

- **Source:** UCI Heart Disease Dataset (available on [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data))
- **Features include:** 
  - Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, ST Depression, Slope, Number of Major Vessels, Thalassemia
- **Target:** Presence of heart disease (`has_disease` 0 = No, 1 = Yes)

---

## Data Preparation

- Handled missing values for numeric and categorical columns
- Converted boolean columns (`fbs`, `exang`) to integers
- Created a **binary target variable** from original numeric labels
- Standardized numeric features and applied **one-hot encoding** to categorical features

---

## Exploratory Data Analysis (EDA)

- Visualized **target distribution** and **age distribution** by heart disease
- Analyzed **proportion of heart disease by chest pain type**
- Created a **correlation heatmap** for numeric features
- Count plots for categorical features like sex

---

## Models Used

1. **Logistic Regression**
   - Evaluated with **accuracy, confusion matrix, classification report, ROC-AUC**
   - Feature importance analyzed using model coefficients

2. **Decision Tree Classifier**
   - Evaluated similarly
   - Provides insight into non-linear patterns

**ROC Curve Comparison** shows Logistic Regression has superior AUC and is the recommended model.

---

## Feature Importance

- Top factors affecting heart disease prediction:
  1. Chest Pain Type (`cp`)
  2. Max Heart Rate Achieved (`thalch`)
  3. ST Depression (`oldpeak`)
  4. Exercise-Induced Angina (`exang`)
  5. Age

---

## Deployment

- The trained Logistic Regression model is saved as `heart_disease_model.pkl`
- Web UI built using **Streamlit**:
  - Users input the **most important factors**
  - Predicts probability of heart disease
  - Modern UI with **colored risk indicators**
  
**Run the app:**

streamlit run app.py
