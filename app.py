import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

st.title("🏡 Real Estate Price Prediction Form")

# Load dataset
DATA_PATH = "data/final.csv"
if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please ensure 'data/final.csv' exists in the 'data' folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Drop unwanted columns
drop_cols = [col for col in df.columns if 'recession' in col.lower() or 'popular' in col.lower() or 'age' in col.lower()]
df = df.drop(columns=drop_cols, errors='ignore')

# Prepare model
X = df.drop('price', axis=1)
y = df['price']
model = LinearRegression()
model.fit(X, y)

feature_names = X.columns.tolist()

# Input Form
st.subheader("Enter Property Details:")

input_data = {}
for feature in feature_names:
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

# Predict
if st.button("Predict Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"🏷️ Estimated Price: ${prediction:,.2f}")
