import streamlit as st
import os
from modules.data_loader import load_data
from modules.model_trainer import train_model

st.title("🏠 Real Estate Price Prediction App")

st.markdown("Upload your real estate dataset (`final.csv`)")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # ✅ Ensure `data/` folder exists
    os.makedirs("data", exist_ok=True)

    # ✅ Save uploaded file to disk
    with open("data/final.csv", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded and saved successfully!")

    # ✅ Load and show data
    df = load_data()
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ✅ Train model
    if st.button("🚀 Train Model"):
        model, mse, r2 = train_model(df)
        st.success(
            f"""✅ **Model Trained Successfully!**  
            
🔢 **Mean Squared Error:** {mse:.2f}  
📈 **R² Score:** {r2:.2f}"""
        )
