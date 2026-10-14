import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction")

cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])

if st.button("Predict"):
    features = np.array([[cp, thalach, oldpeak, exang]])
    result = model.predict(features)[0]

    if result == 1:
        st.error("Heart disease ka risk hai")
    else:
        st.success("Heart disease ka risk nahi hai")
