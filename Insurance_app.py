import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the saved model, encoder, and scaler
MODEL_FILE = "Random Forest_model.joblib"
ENCODER_FILE = "onehot_encoder.joblib"
SCALER_FILE = "scaler.joblib"

model = load(MODEL_FILE)
encoder = load(ENCODER_FILE)
scaler = load(SCALER_FILE)

# Streamlit app title
st.title('Insurance Purchase Prediction')

# Streamlit app description
st.write("""
This app predicts whether a customer is likely to purchase insurance based on their profile information. 
Fill in the details below to make a prediction!
""")

# Input fields for user
gender = st.selectbox('Gender', ['Male', 'Female'])
vehicle_age = st.selectbox('Vehicle Age', ['0-1 Year', '1-2 Year', 'More than 2 Years'])
vehicle_damage = st.selectbox('Vehicle Damage', ['Yes', 'No'])
age = st.slider('Age', 18, 100, 30)
driving_license = st.selectbox('Has Driving License?', ['Yes', 'No'])
region_code = st.selectbox('Region Code', [1, 2, 3, 4, 5])  # Adjust this based on your data
previously_insured = st.selectbox('Previously Insured', ['Yes', 'No'])
annual_premium = st.number_input('Annual Premium', min_value=1000, max_value=100000, value=5000)
policy_sales_channel = st.selectbox('Policy Sales Channel', [1, 2, 3, 4, 5, 6])  # Adjust based on your dataset
vintage = st.number_input('Vintage (years)', min_value=1, max_value=365, value=10)

# Prepare input data for prediction
input_data = {
    'Gender': [gender],
    'Age': [age],
    'Driving_License': [1 if driving_license == 'Yes' else 0],
    'Region_Code': [region_code],
    'Previously_Insured': [1 if previously_insured == 'Yes' else 0],
    'Vehicle_Age': [vehicle_age],
    'Vehicle_Damage': [vehicle_damage],
    'Annual_Premium': [annual_premium],
    'Policy_Sales_Channel': [policy_sales_channel],
    'Vintage': [vintage]
}

# Convert input data to a DataFrame
input_df = pd.DataFrame(input_data)

# Button to trigger prediction
if st.button('Predict Insurance Purchase'):
    # Apply OneHotEncoder to categorical columns
    categorical_columns = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    X_encoded = encoder.transform(input_df[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)

    # Scale numerical columns
    numerical_columns = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
    X_scaled = scaler.transform(input_df[numerical_columns])
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_columns)

    # Combine processed features
    X_final = pd.concat([X_encoded_df, X_scaled_df, input_df[['Driving_License', 'Previously_Insured']]], axis=1)

    # Predict using the trained model
    prediction = model.predict(X_final)

    # Display prediction result
    if prediction[0] == 1:
        st.success('The customer is likely to purchase insurance.')
    else:
        st.warning('The customer is unlikely to purchase insurance.')
