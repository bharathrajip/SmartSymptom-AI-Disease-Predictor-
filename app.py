import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('disease_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Title
st.title("Disease Prediction App")

# Input fields
symptoms = ['None', 'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
            'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
            'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination',
            'fatigue', 'weight_gain']  # example symptoms

input_data = []
for i in range(1, 18):
    symptom = st.selectbox(f"Symptom {i}", symptoms, key=f"symptom_{i}")
    input_data.append(symptom)

# Predict button
if st.button("Predict Disease"):
    input_df = pd.DataFrame([input_data], columns=[f"Symptom_{i}" for i in range(1, 18)])
    for column in input_df.columns:
        input_df[column] = label_encoders[column].transform(input_df[column])
    prediction = model.predict(input_df)
    disease = label_encoders['Disease'].inverse_transform(prediction)
    st.success(f"Predicted Disease: {disease[0]}")