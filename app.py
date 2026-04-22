import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")
st.write("Ingresa los datos del paciente:")

# Inputs
age = st.number_input("Edad", min_value=0, max_value=120, value=30)
avg_glucose_level = st.number_input("Nivel de glucosa", value=100.0)

hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])
ever_married = st.selectbox("¿Ha estado casado/a?", ["No", "Sí"])

smoking_status = st.selectbox(
    "Estado de fumador",
    ["formerly smoked", "never smoked", "Unknown", "smokes"]
)

# Botón
if st.button("Predecir"):

    # Crear DataFrame EXACTO del modelo
    input_data = pd.DataFrame(columns=variables)
    input_data.loc[0] = [0] * len(variables)

    # Numéricas
    if 'age' in variables:
        input_data.at[0, 'age'] = age

    if 'avg_glucose_level' in variables:
        input_data.at[0, 'avg_glucose_level'] = avg_glucose_level

    # Binarias
    if hypertension == "Sí" and 'hypertension_Yes' in variables:
        input_data.at[0, 'hypertension_Yes'] = 1

    if heart_disease == "Sí" and 'heart_disease_Yes' in variables:
        input_data.at[0, 'heart_disease_Yes'] = 1

    if ever_married == "Sí" and 'ever_married_Yes' in variables:
        input_data.at[0, 'ever_married_Yes'] = 1

    # Smoking (match automático incluso con comillas raras)
    for col in variables:
        if "smoking_status" in col and smoking_status in col:
            input_data.at[0, col] = 1

    # Asegurar orden correcto
    input_data = input_data[variables]

    # Escalar
    cols_to_scale = ['age', 'avg_glucose_level']
    cols_to_scale = [c for c in cols_to_scale if c in variables]

    if cols_to_scale:
        input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # 🔥 SOLUCIÓN FINAL: convertir a numpy
    input_array = input_data.values

    # Predicción
    prediction = model.predict(input_array)
    resultado = labelencoder.inverse_transform(prediction)

    # Resultado
    if resultado[0] == 1:
        st.error("⚠️ Alto riesgo de ataque al corazón")
    else:
        st.success("✅ Bajo riesgo de ataque al corazón")
