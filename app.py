import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

st.title("🏡 Real Estate Price Prediction Form")

DATA_PATH = "data/final.csv"
try:
    if not os.path.exists(DATA_PATH):
        logging.error("Missing dataset file: data/final.csv")
        st.error("Dataset not found. Please ensure 'data/final.csv' exists in the 'data' folder.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.exception("Unexpected error loading data.")
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Drop unwanted fields
drop_cols = [col for col in df.columns if 'recession' in col.lower() or 'popular' in col.lower() or 'age' in col.lower()]
df.drop(columns=drop_cols, errors='ignore', inplace=True)

try:
    X = df.drop('price', axis=1)
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)
    logging.info("Model trained successfully.")
except Exception as e:
    logging.exception("Model training failed.")
    st.error(f"Error training model: {e}")
    st.stop()

feature_names = X.columns.tolist()
st.subheader("Enter Property Details:")

input_data = {}
for feature in feature_names:
    try:
        if 'year' in feature.lower():
            input_data[feature] = st.number_input(f"{feature}", min_value=1800, max_value=2100, value=2020)
        elif 'bed' in feature.lower() or 'bath' in feature.lower():
            input_data[feature] = st.slider(f"{feature}", min_value=0, max_value=10, value=2)
        elif 'lot' in feature.lower() or 'size' in feature.lower():
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=50000.0, value=1000.0)
        elif 'basement' in feature.lower():
            choice = st.radio(f"{feature}", options=["Yes", "No"])
            input_data[feature] = 1 if choice == "Yes" else 0
        elif 'property_type' in feature.lower():
            property_choice = st.radio(f"{feature}", options=["Bunglow", "Condo"])
            input_data[feature] = 1 if property_choice == "Bunglow" else 0
        else:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    except Exception as fe:
        logging.warning(f"Input field issue with '{feature}': {fe}")
        st.error(f"Issue with input field '{feature}': {fe}")

if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"🏷️ Estimated Price: ${prediction:,.2f}")
        logging.info("Prediction successful.")
    except NotFittedError:
        logging.error("Prediction failed: model not fitted.")
        st.error("Prediction failed: model not trained.")
    except Exception as pe:
        logging.exception("Prediction failed due to unexpected error.")
        st.error(f"Prediction failed: {pe}")
