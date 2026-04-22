import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")

# Inputs
age = st.number_input("Edad")
avg_glucose_level = st.number_input("Nivel de glucosa")

hypertension = st.selectbox("Hipertensión", [0,1])
heart_disease = st.selectbox("Enfermedad cardíaca", [0,1])
ever_married = st.selectbox("Casado", [0,1])

smoking_status = st.selectbox("Fumador", [0,1,2])  
# (ajusta esto si tienes más categorías)

# Crear DataFrame EXACTO
input_data = pd.DataFrame([{
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'avg_glucose_level': avg_glucose_level,
    'smoking_status': smoking_status
}])

# Escalar variables numéricas
input_data[['age','avg_glucose_level']] = scaler.transform(
    input_data[['age','avg_glucose_level']]
)

# Predicción
if st.button("Predecir"):
    prediction = model.predict(input_data)
    resultado = labelencoder.inverse_transform(prediction)

    if resultado[0] == 1:
        st.error("⚠️ Alto riesgo de ataque al corazón")
    else:
        st.success("✅ Bajo riesgo")
