import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")

# Inputs
age = st.number_input("Edad")
avg_glucose_level = st.number_input("Glucosa")

hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Sí"])
ever_married = st.selectbox("Casado", ["No", "Sí"])

smoking_status = st.selectbox(
    "Estado de fumador",
    ["formerly smoked", "never smoked", "Unknown", "smokes"]
)

# Crear diccionario con todas las columnas en 0
input_dict = {col: 0 for col in variables}

# Asignar valores
input_dict["age"] = age
input_dict["avg_glucose_level"] = avg_glucose_level

if hypertension == "Sí":
    input_dict["hypertension_Yes"] = 1

if heart_disease == "Sí":
    input_dict["heart_disease_Yes"] = 1

if ever_married == "Sí":
    input_dict["ever_married_Yes"] = 1

# Smoking (one-hot)
input_dict[f"smoking_status_{smoking_status}"] = 1

# Convertir a DataFrame con orden correcto
input_data = pd.DataFrame([input_dict])[variables]

# Escalar
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
