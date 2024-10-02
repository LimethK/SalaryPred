import streamlit as st
import pickle
import numpy as np

# Load the saved model (make sure the path is correct)
with open('C:\\Users\\user\\OneDrive - Sri Lanka Institute of Information Technology\\Desktop\\FDM min Project\\mlapp\\best_gb_model.pkl', 'rb') as file:
    best_gb_model = pickle.load(file)

# Streamlit app title
st.title("Machine Learning Model Prediction Web App")

# Provide descriptions for each feature
st.write("Enter the input features for prediction:")

# Feature names corresponding to the 42 columns
feature_names = [
    'Country', 'YearsCodePro', 'WorkExp', 'OpSysProfessional use', 'ProfessionalTech', 'DevType', 'Industry', 'EdLevel', 'Age', 'LanguageHaveWorkedWith', 'RemoteWork', 'Employment', 'ToolsTechHaveWorkedWith', 'DatabaseHaveWorkedWith', 'WebframeHaveWorkedWith'
]

# Create 15 input features for the model based on the feature names
features = []
for feature_name in feature_names:
    feature_value = st.number_input(f'{feature_name}', value=0.0)
    features.append(feature_value)

# Convert the list to a numpy array
features = np.array([features])

# Predict button
if st.button('Predict'):
    # Generate prediction
    prediction = best_gb_model.predict(features)
    # Display the prediction result
    st.write(f"Prediction: {prediction[0]}")
