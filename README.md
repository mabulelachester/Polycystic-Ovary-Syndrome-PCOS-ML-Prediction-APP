## Polycystic-Ovary-Syndrome-PCOS-ML-Prediction-APP
This project is a machine learning-based web application for predicting Polycystic Ovary Syndrome (PCOS) based on key health indicators. It utilizes a trained model deployed using Streamlit and provides an easy-to-use interface for users to input their health data and receive predictions.

# Dataset
The dataset consists of 1,000 patient records with the following key features:

- **Age (years):** Ranges from 18 to 45.
- **BMI (kg/m²):** Body mass index ranging from 18 to 35.
- **Menstrual Irregularity (binary):** 0 = No, 1 = Yes.
- **Testosterone Level (ng/dL):** Ranges from 20 to 100.
- **Antral Follicle Count:** Ranges from 5 to 30.
- **PCOS Diagnosis (binary):** 0 = No PCOS, 1 = PCOS detected.

# Features
- PCOS Diagnosis Prediction: Uses a trained machine learning model.
- User-friendly Web Interface: Built with Streamlit.
- Automated Data Scaling: Ensures input data is processed in the same way as training data.

# Installation
To run the app locally, follow these steps:

- Prerequisites
- Python 3.8+
- Streamlit
- NumPy
- Scikit-learn

# Steps
1. Clone the repository:
   - git clone https://github.com/mabulelachester/PCOS-Prediction-App.git
   - cd PCOS-Prediction-App
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py

# PCOS Prediction App

Below is the main application code from `app.py`:

```python
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
```

# Usage

1. Enter the required health parameters in the web app.
2. Click the Predict button.
3. View the PCOS diagnosis and probability score.

# Contributors
Mabulela Chester- Developer and Maintainer
