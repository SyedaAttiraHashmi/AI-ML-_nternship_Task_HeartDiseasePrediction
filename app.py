import streamlit as st
import pandas as pd
import joblib

# Load the trained model
heart_model = joblib.load('heart_disease_model.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#e63946;'>❤️ Heart Disease Prediction App</h1>
    <p style='text-align:center; color:white;'>Enter the most important health factors to predict heart disease risk.</p>
    """,
    unsafe_allow_html=True
)

# Sidebar Inputs: Only important factors
st.sidebar.header("Key Patient Information")
age = st.sidebar.number_input("Age", 20, 100, 50)
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
thalch = st.sidebar.number_input("Max Heart Rate Achieved", 70, 220, 150)
oldpeak = st.sidebar.number_input("ST depression induced by exercise", 0.0, 10.0, 1.0)
exang = st.sidebar.selectbox("Exercise Induced Angina", [True, False])

# Prepare input dataframe
input_data = pd.DataFrame({
    'age':[age],
    'cp':[cp],
    'thalch':[thalch],
    'oldpeak':[oldpeak],
    'exang':[exang],
    
    # Fill remaining features with typical/median values
    'sex':['Male'],            # default
    'trestbps':[130],          # median
    'chol':[240],              # median
    'fbs':[0],
    'restecg':['normal'],
    'slope':['flat'],
    'ca':[0],
    'thal':['normal']
})

# Prediction
if st.button("Predict"):
    prediction = heart_model.predict(input_data)[0]
    prediction_prob = heart_model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    if prediction == 1:
        col1.metric("Prediction", "Heart Disease", delta=f"Risk: {prediction_prob*100:.2f}%", delta_color="inverse")
        col2.markdown(
            f"<div style='padding:20px; background-color:#e63946; color:white; text-align:center; border-radius:10px;'>"
            f"<h3>⚠ High Risk: {prediction_prob*100:.2f}% probability</h3></div>",
            unsafe_allow_html=True
        )
    else:
        col1.metric("Prediction", "No Heart Disease", delta=f"Confidence: {(1-prediction_prob)*100:.2f}%", delta_color="normal")
        col2.markdown(
            f"<div style='padding:20px; background-color:#2a9d8f; color:white; text-align:center; border-radius:10px;'>"
            f"<h3>✅ Low Risk: {(1-prediction_prob)*100:.2f}% confidence</h3></div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:white;'>This prediction is for educational purposes only and should not replace professional medical advice.</p>",
        unsafe_allow_html=True
    )
