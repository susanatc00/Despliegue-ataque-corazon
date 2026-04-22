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

# Botón de predicción
if st.button("Predecir"):

    # Crear diccionario con TODAS las columnas en 0
    input_dict = dict.fromkeys(variables, 0)

    # Variables numéricas
    input_dict['age'] = age
    input_dict['avg_glucose_level'] = avg_glucose_level

    # Variables binarias
    if hypertension == "Sí":
        input_dict['hypertension_Yes'] = 1

    if heart_disease == "Sí":
        input_dict['heart_disease_Yes'] = 1

    if ever_married == "Sí":
        input_dict['ever_married_Yes'] = 1

    # 🔥 Smoking dinámico (evita errores por comillas)
    for col in variables:
        if "smoking_status" in col and smoking_status in col:
            input_dict[col] = 1

    # Crear DataFrame con orden EXACTO
    input_data = pd.DataFrame([input_dict])[variables]

    # Escalar variables numéricas
    cols_to_scale = ['age', 'avg_glucose_level']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Predicción
    prediction = model.predict(input_data)
    resultado = labelencoder.inverse_transform(prediction)

    # Resultado
    if resultado[0] == 1:
        st.error("⚠️ Alto riesgo de ataque al corazón")
    else:
        st.success("✅ Bajo riesgo de ataque al corazón")
