import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")

# Inputs (ajusta a TODAS tus variables)
age = st.number_input("Edad")
avg_glucose_level = st.number_input("Glucosa")

hypertension = st.selectbox("Hipertensión", [0,1])
heart_disease = st.selectbox("Enfermedad cardíaca", [0,1])
ever_married = st.selectbox("Casado", [0,1])

# IMPORTANTE: DataFrame con TODAS las columnas
input_dict = {col: 0 for col in variables}

# Asignar valores reales
input_dict["age"] = age
input_dict["avg_glucose_level"] = avg_glucose_level
input_dict["hypertension_1"] = hypertension
input_dict["heart_disease_1"] = heart_disease
input_dict["ever_married_Yes"] = ever_married

# Convertir a DataFrame
input_data = pd.DataFrame([input_dict])

# Escalar
input_data[['age','avg_glucose_level']] = scaler.transform(
    input_data[['age','avg_glucose_level']]
)

# Predicción
if st.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = labelencoder.inverse_transform(prediction)

    if resultado[0] == 1:
        st.error("⚠️ Alto riesgo")
    else:
        st.success("✅ Bajo riesgo")
