import os
import requests
import streamlit as st
import pickle
import pandas as pd
from helpers import (
    CreateHasIncomeColumn,
    CreateProvidedRiskScoreColumn,
    ApplyMapToLoanTitles,
    ApplyLabelToEmploymentLength,
    title_mapping,
    empl_length_mapping,
)

# Setup environment credentials (you'll need to change these)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json" # change for your GCP key
PROJECT = "daniels-dl-playground" # change for your GCP project
REGION = "us-central1" # change for your GCP region (where your model is hosted)

# Get feat_eng pipelines
current_directory = os.getcwd()

file_name = 'feat_eng_pipe-0.1.0.pkl'
file_path = os.path.join(current_directory, file_name)
with open(file_path, 'rb') as file:
    feat_eng_pipe = pickle.load(file)

# Get preprocessing pipe
file_name = 'preprocess_pipe-0.1.0.pkl'
file_path = os.path.join(current_directory, file_name)
with open(file_path, 'rb') as file:
    preprocess_pipe = pickle.load(file)

# Load model
file_name = 'dt_model-0.1.0.pkl'
file_path = os.path.join(current_directory, file_name)
with open(file_path, 'rb') as file:
    model = pickle.load(file)


# Streamlit app
st.title("Loan Application Prediction App")
st.header("Input Features")

#Input data
amount_requested = st.number_input("Amount Requested", min_value=1000, step=1)
loan_title = st.selectbox("Loan Title", list(title_mapping.keys()))
risk_score = st.number_input("Risk Score", min_value=0, max_value=990, step=1)
dti = st.number_input("Debt-To-Income Ratio", min_value=0.0, step=0.1)
employment_length = st.selectbox("Employment Length", list(empl_length_mapping.keys()))

# Create a DataFrame with the input features
input_data = pd.DataFrame({
    "Amount_Requested": [amount_requested],
    "Loan_Title": [loan_title],
    "Risk_Score": [risk_score],
    "Debt-To-Income_Ratio": [dti],
    "Employment_Length": [employment_length],
})

# Define preprocess function
def preprocess_data(df):
    df = feat_eng_pipe.transform(df)
    df = preprocess_pipe.transform(df)
    return df

def make_prediction(df):
    """ Takes user input data and returns if the loan will be accepted or rejected. """
    
    df_preprocessed = preprocess_data(df)
    prediction = model.predict(df_preprocessed)[0]
    return prediction

# Display prediction
user_prediction = make_prediction(input_data)
st.header("Prediction")
st.write("Loan Application Status:", "Accepted" if user_prediction == 1 else "Rejected")
