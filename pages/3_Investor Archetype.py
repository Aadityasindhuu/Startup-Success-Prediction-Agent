#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# === Set Page Config ===
st.set_page_config(page_title="Investor Archetype Agent", layout="wide")

# === Background Image Setup ===
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
            background-position: center;
            background-repeat: no-repeat;
            color: #FFD700 !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.15) !important;
        }}
        .stTextInput > div > div > input,
        textarea, .stTextArea {{
            background-color: rgba(255,255,255,0.05) !important;
            color: #FFD700 !important;
            border: 1px solid #FFD700;
            border-radius: 10px;
        }}
        .stButton > button {{
            background-color: rgba(255, 215, 0, 0.1);
            color: #FFD700;
            border: 1px solid #FFD700;
            border-radius: 10px;
            font-weight: bold;
        }}
        .stButton > button:hover {{
            background-color: rgba(255, 215, 0, 0.3);
            transition: 0.3s ease-in-out;
            color: black;
        }}
        .stMarkdown {{
            background-color: rgba(0, 0, 0, 0.25);
            padding: 10px;
            border-radius: 10px;
            color: #FFD700 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your background image path here
set_bg("pages/Investor Archetype bg.png")

# === App Title ===
st.title("🔍 Investor Archetype Agent")
st.markdown("Analyze and cluster investor behavior based on deal activity and investment traits.")

# === Load Clustered Data and Models ===
try:
    df = pd.read_csv("models/investor_clusters.csv")
    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.subheader("📊 Investor Behavioral Clusters")
    st.write("Below is the categorized behavior of investors based on their investment frequency, size, and risk profiles.")

    # Show table
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=df, x='TotalInvestment', y='AvgInvestment', hue='Cluster', palette='cool', s=100)
    plt.title("Investor Clusters: Total vs. Average Investment", fontsize=16)
    st.pyplot(fig)

    # Investor Search
    st.subheader("🔎 Find Investor Cluster")
    investor_name = st.text_input("Enter investor name:")

    if investor_name:
        result = df[df['Investor'].str.lower().str.contains(investor_name.lower())]
        if not result.empty:
            st.success(f"Cluster Info for '{investor_name}':")
            st.dataframe(result)
        else:
            st.warning("Investor not found in clusters.")

except FileNotFoundError:
    st.error("Required model files or CSV missing. Please run backend clustering step first.")


