import streamlit as st
import requests
import pandas as pd
import json
import re
   
API_URL = "http://localhost:8000"  

models = []

# Set the title and caption for the Streamlit app
st.set_page_config(page_title="ECG analysis")
st.title("ECG analysis")
st.caption(
    "App integrated with a pretrained model for ECG data classification with multiple user input options"
)


if st.button("get_models"):

    response = requests.get(f"{API_URL}/models")

    if response.status_code == 200:
        st.success("models fetched successfully")
        models = st.write(response.json()['models'])
    else:
        st.write(response.status_code)
        st.write(response.reason)
        st.error("Failed to fetch models.")

# Set active model
st.header("Set Active Model")
model_id = st.selectbox("Select Model ID", [model["id"] for model in models])
if st.button("Set Active Model"):
    response = requests.post(f"{API_URL}/set", json={"model_id": model_id})
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error("Failed to set active model.")

def extract_data(input_string):
    # Regular expression pattern to match float numbers
    float_pattern = r'-?\d+\.\d+'

    # Find all matches in the input string
    float_numbers = re.findall(float_pattern, input_string)

    # Convert the matched strings to float numbers
    float_numbers = [float(num) for num in float_numbers]

    return float_numbers

st.header("Make Prediction")
data = st.text_input("Enter input data (JSON format)", "[]")

if st.button("Predict"):
    data = json.loads(data)
    response = requests.post(f"{API_URL}/predict", json={"data": data})
    if response.status_code == 200:
        st.success(f"Prediction: {response.json()['prediction']}")
    elif json.JSONDecodeError:
        st.error("Invalid JSON format.")
    else:
        st.error("Prediction failed.")
    
st.header("Train Model")
params = st.text_input("Enter training parameters (JSON format)", "{}")
if st.button("Train Model"):
    try:
        params_dict = json.loads(params)
        response = requests.post(f"{API_URL}/fit", json={"params": params_dict})
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error("Training failed.")
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
