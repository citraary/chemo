 
import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("tumor_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Tumor Size Shrinkage Prediction")
st.write("Predict tumor size shrinkage based on CEA and MTHFR values.")

# Input fields
cea = st.number_input("Enter CEA Value:", min_value=0.0, step=0.01)
mthfr = st.number_input("Enter MTHFR Value:", min_value=0.0, step=0.01)

if st.button("Predict"):
    # Scale input values
    input_data = np.array([[cea, mthfr]])
    scaled_data = scaler.transform(input_data)

    # Predict tumor shrinkage
    prediction = model.predict(scaled_data)[0]

    # Display the result
    st.write(f"Predicted Tumor Size Shrinkage: {prediction:.2f}%")
