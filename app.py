import streamlit as st
import pickle
import pandas as pd

# Cargar modelo
model, labelencoder, variables, scaler = pickle.load(open("modelo-class.pkl", "rb"))

st.title("Predicción de Ataque al Corazón ❤️")

# Inputs
age = st.number_input("Edad", 0, 120, 30)
glucose = st.number_input("Glucosa", value=100.0)

hypertension = st.selectbox("Hipertensión", ["No","Sí"])
heart = st.selectbox("Enfermedad cardíaca", ["No","Sí"])
married = st.selectbox("Casado", ["No","Sí"])

smoke = st.selectbox("Fumador",
    ["formerly smoked","never smoked","Unknown","smokes"]
)

if st.button("Predecir"):

    # 🔥 crear vector EXACTO del tamaño del modelo
    fila = [0]*len(variables)

    # llenar por índice (NO por nombre → evita todos los errores)
    for i,col in enumerate(variables):

        if col == "age":
            fila[i] = age

        elif col == "avg_glucose_level":
            fila[i] = glucose

        elif col == "hypertension_Yes" and hypertension=="Sí":
            fila[i] = 1

        elif col == "heart_disease_Yes" and heart=="Sí":
            fila[i] = 1

        elif col == "ever_married_Yes" and married=="Sí":
            fila[i] = 1

        elif "smoking_status" in col and smoke in col:
            fila[i] = 1

    # convertir a DataFrame
    X = pd.DataFrame([fila], columns=variables)

    # escalar
    if "age" in variables and "avg_glucose_level" in variables:
        X[["age","avg_glucose_level"]] = scaler.transform(
            X[["age","avg_glucose_level"]]
        )

    # numpy (sin validaciones de sklearn)
    pred = model.predict(X.values)
    res = labelencoder.inverse_transform(pred)

    if res[0]==1:
        st.error("⚠️ Alto riesgo")
    else:
        st.success("✅ Bajo riesgo")
