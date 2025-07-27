import streamlit as st
import joblib
import base64


# === Load model and vectorizer ===
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
inv_label_map = {0: "Weak", 1: "Neutral", 2: "Strong"}

# === Page setup ===
st.set_page_config(page_title="Pitch Strength Classifier", layout="wide")

# === Set background image ===
def set_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Sidebar transparent */
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}

        /* Text input + button transparent */
        textarea, .stTextArea, .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 10px;
            border: 1px solid #aaa;
        }}

        button[kind="primary"] {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border: 1px solid #aaa;
        }}

        /* Output text background */
        .stMarkdown {{
            background-color: rgba(0, 0, 0, 0.25) !important;
            color: white !important;
            padding: 10px;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Apply the correct background image
set_bg("pages/Pitch analysis bg.png")

# === App title and description ===
st.markdown("<h1 style='text-align:center; color:white;'> Pitch Strength Classifier</h1>", unsafe_allow_html=True)

# === Spacing to push content down visually
st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# === Input & Prediction Section
with st.container():
    st.markdown("<h4 style='color:white;'>💬 Enter your startup pitch here:</h4>", unsafe_allow_html=True)
    user_input = st.text_area("", height=150, key="pitch_input", label_visibility="collapsed")

    if st.button("🔍 Predict Pitch Strength"):
        if user_input.strip():
            tfidf_input = vectorizer.transform([user_input])
            prediction = model.predict(tfidf_input)[0]
            prediction_label = inv_label_map[prediction]
            st.markdown(f"<b>🧠 Prediction:</b> {prediction_label}", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#ff4444;'>⚠️ Please enter a pitch before predicting.</span>", unsafe_allow_html=True)
4

# === Load model and vectorizer ===
model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
inv_label_map = {0: "Weak", 1: "Neutral", 2: "Strong"}

# === Page setup ===
st.set_page_config(page_title="Pitch Strength Classifier", layout="wide")

# === Set background image ===
def set_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>0
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Sidebar transparent */
        section[data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0) !important;
        }}

        /* Text input + button transparent */
        textarea, .stTextArea, .stTextInput > div > div > input {{
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: white !important;
            border-radius: 10px;
            border: 1px solid #aaa;
        }}

        button[kind="primary"] {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border: 1px solid #aaa;
        }}

        /* Output text background */
        .stMarkdown {{
            background-color: rgba(0, 0, 0, 0.25) !important;
            color: white !important;
            padding: 10px;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("pages/Pitch analysis bg.png")

# === App title and description ===
st.markdown("<h1 style='text-align:center; color:white;'> Pitch Strength Classifier</h1>", unsafe_allow_html=True)


# === Spacing to push to bottom visually
st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# === Input & Prediction Section (at bottom, transparent)
with st.container():
    st.markdown("<h4 style='color:white;'>💬 Enter your startup pitch here:</h4>", unsafe_allow_html=True)
    user_input = st.text_area("", height=150, key="pitch_input", label_visibility="collapsed")

    if st.button("🔍 Predict Pitch Strength"):
        if user_input.strip():
            tfidf_input = vectorizer.transform([user_input])
            prediction = model.predict(tfidf_input)[0]
            prediction_label = inv_label_map[prediction]
            st.markdown(f"<b>🧠 Prediction:</b> {prediction_label}", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#ff4444;'>⚠️ Please enter a pitch before predicting.</span>", unsafe_allow_html=True)

           



