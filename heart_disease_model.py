

#Import Libraries


import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Basic plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

#Read the Dataset
data_heart = pd.read_csv("data\heart.csv")
data_heart.head(10)

"""# 2. Data cleaning"""

print("Initial shape:", data_heart.shape)

#  Check missing values
missingval = data_heart.isnull().sum()
print("\nMissing values per column:")

missingval

print(missingval[missingval > 0] if missingval.sum() > 0 else "None")

# Fill numeric columns with median
numeric_cols = data_heart.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    data_heart[numeric_cols] = data_heart[numeric_cols].fillna(data_heart[numeric_cols].median())
    print("➡ Filled missing numeric values with median.")

# Drop duplicates
duplicate_count = data_heart.duplicated().sum()
if duplicate_count > 0:
    print(f"➡ Found {duplicate_count} duplicate rows. Dropping...")
    df = data_heart.drop_duplicates()
else:
    print("No duplicate rows found.")

print("Shape after cleaning:", data_heart.shape)

# Columns that are categorical and have missing values
cat_col = ["fbs", "restecg", "exang", "slope", "thal"]
# Keep only the ones that actually exist in the dataframe
cat_col = [col for col in cat_col if col in data_heart.columns]

print("\nCategorical columns with missing values:", cat_col)

# Fill missing categorical values with mode
for col in cat_col:
    val_mode = data_heart[col].mode(dropna=True)[0]
    data_heart[col] = data_heart[col].fillna(val_mode)
    print(f"➡ Filled missing values in '{col}' with mode: {val_mode}")

#  Check missing values
missingval = data_heart.isnull().sum()
missingval

"""## 3. Prepare target from `num`"""

# Convert 0..4 into binary
data_heart["target"] = (data_heart["num"] > 0).astype(int)

print("Target value counts (0 = no disease, 1 = disease):")
print(data_heart["target"].value_counts())

data_heart

# Drop original num if we don't want it as a feature
data_heart = data_heart.drop(columns=["num"])

data_heart

"""#EDA"""

data_heart.info()

data_heart.describe()



## Target distribution plot
plt.figure()
data_heart["target"].value_counts().plot(kind="bar")
plt.title("Target Distribution (Heart Disease)")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.tight_layout()

# Age distribution by heart disease
plt.figure()
sns.histplot(data=data_heart, x='age', hue='target', bins=20, kde=True, palette=['skyblue', 'salmon'])
plt.title("Age Distribution by Heart Disease Status")
plt.xlabel("Age")
plt.ylabel("Count")

#Is there an association between chest pain type and the presence of heart disease?
# Create binary target
data_heart['has_disease'] = (data_heart['target'] > 0).astype(int)

# Group by chest pain type and calculate mean (proportion with disease)
cp_group = data_heart.groupby('cp')['has_disease'].mean().sort_values()

plt.figure(figsize=(8, 5))
plt.bar(cp_group.index, cp_group.values)
plt.xticks(rotation=20)
plt.ylabel('Proportion with Heart Disease')
plt.xlabel('Chest Pain Type')
plt.title('Heart Disease Proportion by Chest Pain Type')
plt.tight_layout()
plt.show()

"""The boxplots show that resting blood pressure and cholesterol have very similar distributions in patients with and without heart disease. Median values and overall ranges overlap substantially, with high outliers present in both groups. This suggests that, in this dataset, neither resting blood pressure nor total cholesterol alone clearly differentiates between patients with and without heart disease."""

# --------- Visual 1: Heatmap of correlations ---------
plt.figure(figsize=(10, 8))
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'target']
corr = data_heart[numeric_features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# --------- Visual 2: Bar plot of heart disease by sex ---------
plt.figure(figsize=(6, 5))
sns.countplot(x='sex', hue='target', data=data_heart, palette='Set2')
plt.title("Heart Disease Counts by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Heart Disease (num)")
plt.show()

"""#Data Prepration|

"""

data_heart.head(10)

data_heart = data_heart.drop(columns=["target"])

data_heart

# Convert boolean columns to integers
data_heart['fbs'] = data_heart['fbs'].astype(int)
data_heart['exang'] = data_heart['exang'].astype(int)

data_heart

# Features and target
X_heart = data_heart.drop(columns=['id', 'dataset', 'has_disease'])
Y_heart = data_heart['has_disease']

# Identify categorical and numeric features
num_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
cat_features = ['sex', 'cp', 'restecg', 'slope', 'thal']

# Preprocessing & Split
# --------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

X_heart_train, X_heart_test, Y_heart_train, Y_heart_test = train_test_split(X_heart, Y_heart, test_size=0.2, random_state=42)

# Logistic Regression Model

logreg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
logreg_pipeline.fit(X_heart_train, Y_heart_train)

# Predict & Probabilities
y_pred_logistic = logreg_pipeline.predict(X_heart_test)
y_prob_logistic = logreg_pipeline.predict_proba(X_heart_test)[:,1]

# Logistic Regression Evaluation

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(Y_heart_test, y_pred_logistic))
print("Confusion Matrix:\n", confusion_matrix(Y_heart_test, y_pred_logistic))
print("Classification Report:\n", classification_report(Y_heart_test, y_pred_logistic))
print("ROC-AUC:", roc_auc_score(Y_heart_test, y_prob_logistic))

# ROC Curve
fpr, tpr, _ = roc_curve(Y_heart_test, y_prob_logistic)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(Y_heart_test, y_prob_logistic)))
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()

"""The Area Under the Curve (AUC) for this Logistic Regression model is $0.88$, which measures the two-dimensional area beneath the ROC curve. This high value indicates that there is an $88\%$ probability the model will rank a randomly chosen positive case higher than a randomly chosen negative case, signifying strong discriminatory power. Since the curve bows significantly toward the top-left corner and maintains clear separation from the random baseline (the diagonal line, which represents an AUC of $0.5$), the Logistic Regression model performs very well on this classification task, reliably achieving high True Positive Rates while minimizing False Positive classifications."""

# Feature Importance (Coefficients)
# --------------------------
# Extract feature names after one-hot encoding
ohe = logreg_pipeline.named_steps['preprocess'].transformers_[1][1]  # OneHotEncoder
ohe_features = list(ohe.get_feature_names_out(cat_features))
all_features = num_features + ohe_features

coeff = logreg_pipeline.named_steps['classifier'].coef_[0]
impt_features = pd.Series(coeff, index=all_features).sort_values(key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=impt_features.values, y=impt_features.index)
plt.title("Top Feature Influence (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

# Decision Tree Model

dt_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])
dt_pipeline.fit(X_heart_train, Y_heart_train)

y_pred_dt = dt_pipeline.predict(X_heart_test)
y_prob_dt = dt_pipeline.predict_proba(X_heart_test)[:,1]

# Evaluation
print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(Y_heart_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(Y_heart_test, y_pred_dt))
print("Classification Report:\n", classification_report(Y_heart_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(Y_heart_test, y_prob_dt))

# ROC Curve comparison
fpr_dt, tpr_dt, _ = roc_curve(Y_heart_test, y_prob_dt)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(Y_heart_test, y_prob_logistic)))
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = {:.2f})'.format(roc_auc_score(Y_heart_test, y_prob_dt)))
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Comparison')
plt.legend()
plt.show()

"""Based on both the superior AUC score (0.88 vs. 0.79) and the visual dominance of the blue curve over the orange curve across nearly all thresholds, the Logistic Regression model is the better-performing classifier for this specific dataset and prediction task."""

# Save the trained Logistic Regression pipeline
import joblib

joblib.dump(logreg_pipeline, 'heart_disease_model.pkl')
print("Model saved as heart_disease_model.pkl")
