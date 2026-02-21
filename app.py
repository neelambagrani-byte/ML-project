import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")

# User Inputs
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.number_input("Resting ECG (0-2)", 0, 2)
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak")
slope = st.number_input("Slope (0-2)", 0, 2)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4)
thal = st.number_input("Thal (0-3)", 0, 3)

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")
