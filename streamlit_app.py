import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# === Page Config ===
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

# === Title ===
st.title("🧬 Breast Cancer Diagnosis - Upload Your Data")

# === Sidebar ===
st.sidebar.header("Upload Settings")

# === Load Model and Scaler ===
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("attention_model.h5", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# === File Upload ===
st.subheader("📄 Upload Your CSV (Single Row Only)")
uploaded_file = st.file_uploader("Upload a CSV file with 1 row (30 features only)", type=["csv"])

# === Read and Validate Uploaded File ===
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(input_df)

        if input_df.shape[0] != 1:
            st.warning("⚠️ Please upload exactly one row of data.")
        else:
            st.success("Data looks good! Proceeding to prediction step...")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
