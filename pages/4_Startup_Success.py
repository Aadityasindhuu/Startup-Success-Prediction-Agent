#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="🚀 Startup Success Predictor", layout="centered")

# Custom CSS for premium styling
st.markdown("""
    <style>
        html, body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: #ffffff;
        }
        .reportview-container {
            padding-top: 2rem;
        }
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #fddb3a;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #ccc;
        }
        .stButton>button {
            background-color: #fddb3a;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
        }
        .result-box {
            background-color: #1e1e1e;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 0 15px #fddb3a;
            text-align: center;
            font-size: 1.2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Startup Success Prediction Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Know your odds. Predict your startup’s future with AI.</div>', unsafe_allow_html=True)
st.markdown("")

# Load the trained model
with open("models/success_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the column names used in training
with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Input form
st.subheader("📥 Enter Startup Details")

user_input = {}
cols = st.columns(2)

for i, col in enumerate(model_columns):
    with cols[i % 2]:
        user_input[col] = st.number_input(f"{col.replace('_', ' ').title()}", min_value=0.0)

# Prediction
if st.button("🔮 Predict Success"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    result_text = "🎉 High Chances of Success!" if prediction == 1 else "⚠️ Likely to Fail – Consider Pivoting."

    st.markdown(f"""
        <div class="result-box">
            <strong>Prediction:</strong><br>
            {result_text}
        </div>
    """, unsafe_allow_html=True)

