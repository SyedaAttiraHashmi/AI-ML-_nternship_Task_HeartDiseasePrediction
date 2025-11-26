# Heart Disease Prediction System

A machine learning application for predicting heart disease risk based on health data using Logistic Regression and Decision Tree classifiers.

## Features

- **Data Cleaning**: Handles missing values and duplicates
- **Exploratory Data Analysis (EDA)**: Comprehensive visualizations and statistical analysis
- **Model Training**: Logistic Regression and Decision Tree classifiers
- **Model Evaluation**: Accuracy, ROC-AUC, and Confusion Matrix metrics
- **Feature Importance**: Analysis of important features affecting predictions
- **Streamlit UI**: Interactive web application for predictions and analysis

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Place the Heart Disease UCI dataset file as `heart.csv` in the project root directory. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci).

## Usage

### 1. Train the Models

Run the training script to clean data, perform EDA, train models, and generate evaluation metrics:

```bash
python heart_disease_model.py
```

This will:
- Load and clean the dataset
- Perform exploratory data analysis
- Train Logistic Regression and Decision Tree models
- Evaluate models and generate metrics
- Save models and visualizations

### 2. Launch the Streamlit UI

Start the interactive web application:

```bash
streamlit run app.py
```

The application will open in your browser with the following pages:

- **Home**: Overview and instructions
- **Data Overview**: Dataset statistics and information
- **EDA**: Interactive exploratory data analysis
- **Model Prediction**: Make predictions for new patients
- **Model Evaluation**: View model performance metrics

## Project Structure

```
.
├── heart.csv                    # Dataset file (you need to add this)
├── heart_disease_model.py      # Main training and evaluation script
├── app.py                      # Streamlit UI application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Saved models (created after training)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── scaler.pkl
└── plots/                      # Generated visualizations (created after training)
    ├── target_distribution.png
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    ├── roc_curves.png
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_decision_tree.png
    └── feature_importance.png
```

## Model Evaluation Metrics

The system evaluates models using:

- **Accuracy**: Overall prediction accuracy
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
- **Feature Importance**: Key features affecting predictions

## Features Analyzed

Common features in the heart disease dataset include:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope of Peak Exercise ST
- Number of Major Vessels
- Thalassemia

## Notes

- Ensure the dataset file is named `heart.csv` and placed in the project root
- Run the training script before using the Streamlit app for predictions
- The models are saved automatically after training
- All visualizations are saved in the `plots/` directory

## License

This project is for educational purposes.

