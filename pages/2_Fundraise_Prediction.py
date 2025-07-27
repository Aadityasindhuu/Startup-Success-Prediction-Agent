#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import base64

# === Load Model and Label Encoder ===
model = joblib.load("models/fundraise_model.pkl")
label_encoder = joblib.load("models/fundraise_label_encoder.pkl")

# === Page Setup ===
st.set_page_config(page_title="Fundraise Prediction Agent", layout="wide")

# === Background Setup ===
def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}

        /* Clean white text, no glow */
        h1, h2, h3, h4, h5, h6, label, p, span {{
            color: #ffffff !important;
            text-shadow: none !important;
        }}

        /* Transparent and neat input fields */
        .stTextInput input, .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: #ffffff !important;
            border: 1px solid #ffffff44 !important;
            border-radius: 10px;
            padding: 10px;
        }}

        /* Improved button style */
        button[kind="primary"] {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border: 1px solid #ffffff88 !important;
            font-weight: 500;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            transition: 0.4s ease-in-out;
        }}
        button[kind="primary"]:hover {{
            background-color: #ffffff !important;
            color: #000000 !important;
            box-shadow: 0 0 12px #ffffff88;
        }}

        /* Transparent sidebar */
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.0) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# === Set Background Image ===
set_bg("pages/Fundraise Prediction bg.png")  # Make sure image path is correct

# === Title and Description ===
st.title("💰 Fundraise Prediction Agent")
st.markdown("Predict the **type of investment** your startup may receive based on your profile.")

# === Form Inputs ===
with st.form("fundraise_form"):
    col1, col2 = st.columns(2)

    with col1:
        startup = st.text_input("🔹 Startup Name")
        industry = st.text_input("🔹 Industry Vertical")
        subvertical = st.text_input("🔹 Sub Vertical")

    with col2:
        city = st.text_input("📍 City Location")
        investors = st.text_input("👥 Investors")
        amount = st.text_input("💵 Amount in USD", value="1000000")

    submit = st.form_submit_button("🔍 Predict")

# === Prediction Logic ===
if submit:
    try:
        combined = f"{startup} {industry} {subvertical} {city} {investors} Amount {amount}"
        label = model.predict([combined])[0]
        prediction = label_encoder.inverse_transform([label])[0]
        st.success(f"✅ Predicted Investment Type: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

