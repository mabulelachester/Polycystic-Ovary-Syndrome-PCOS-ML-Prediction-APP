import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open("pcos_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI for the user to input values
st.title("PCOS Prediction")

age = st.number_input("Age", min_value=18, max_value=45, value=30)
bmi = st.number_input("BMI", min_value=18.1, max_value=35.0, value=26.4)
menstrual_irregularity = st.radio("Menstrual Irregularity", (0, 1), index=0)  # 0: No, 1: Yes
testosterone_level = st.number_input("Testosterone Level (ng/dL)", min_value=20.0, max_value=100.0, value=60.0)
antral_follicle_count = st.number_input("Antral Follicle Count", min_value=5, max_value=29, value=18)

# Organize the input features into a list
input_data = [age, bmi, menstrual_irregularity, testosterone_level, antral_follicle_count]

# When the user clicks the predict button
if st.button("Predict"):
    # Convert the input data to a NumPy array and reshape it for prediction
    input_data = np.array(input_data).reshape(1, -1)

    # Apply the scaler (same scaling as during model training)
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    # Display the result
    if prediction == 1:
        st.write("### Prediction: PCOS Detected!")
        st.write(f"Probability: {probability * 100:.2f}%")
    else:
        st.write("### Prediction: No PCOS detected.")
        st.write(f"Probability: {probability * 100:.2f}%")
