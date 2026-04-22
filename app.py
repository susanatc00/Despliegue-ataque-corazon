import streamlit as st
import pickle
import numpy as np

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")

# Inputs (ajusta según tus variables reales)
age = st.number_input("Edad", min_value=0, max_value=120)
glucose = st.number_input("Nivel de glucosa")

hypertension = st.selectbox("Hipertensión", [0,1])
heart_disease = st.selectbox("Enfermedad cardíaca", [0,1])

# Crear array (IMPORTANTE: orden igual al entrenamiento)
input_data = np.array([[age, glucose, hypertension, heart_disease]])

# Escalar (solo si aplica a esas variables)
input_data[:, :2] = scaler.transform(input_data[:, :2])

# Botón
if st.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = labelencoder.inverse_transform(prediction)

    if resultado[0] == 1:
        st.error("⚠️ Alto riesgo de ataque al corazón")
    else:
        st.success("✅ Bajo riesgo")
