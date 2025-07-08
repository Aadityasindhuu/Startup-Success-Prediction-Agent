import streamlit as st
import base64
from PIL import Image

# Load and encode your background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode()
        st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }}

    .css-10trblm, .css-1v3fvcr, .css-qbe2hs, .css-1r6slb0 {{
        color: #ffffff !important;
    }}

    .card {{
        background: rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: #fff;
        margin-bottom: 1.5rem;
    }}

    .hero {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 3rem 1rem;
        border-radius: 20px;
        text-align: center;
        color: #fff;
        margin-bottom: 2rem;
    }}

    h1, h3, p, .stMarkdown, .stCaption {{
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{b64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .card {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 20px;
                padding: 1.5rem;
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                color: #fff;
                margin-bottom: 1.5rem;
            }}
            .hero {{
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                padding: 3rem 1rem;
                border-radius: 20px;
                text-align: center;
                color: #fff;
                margin-bottom: 2rem;
            }}
            h1, h3, p, .stMarkdown, .stCaption {{
                color: #ffffff !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# SET BACKGROUND
set_background("background image.png")

# Hero Section
st.markdown(f"""
<div class="hero">
    <h1> Startup Growth Intelligence Dashboard</h1>
    <p>AI-powered platform to accelerate startup decision-making</p>
</div>
""", unsafe_allow_html=True)

# Agent Cards
st.subheader(" Explore Our AI Agents")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>Pitch Analysis Agent</h3>
        <p>Understand your pitch performance and improve communication with investors.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Fundraising Prediction Agent</h3>
        <p>Predict the likelihood of receiving funding based on startup traits.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>Investor Archetype Agent</h3>
        <p>Profile investors using behavioral clustering to find best-fit VCs.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Startup Success Predictor</h3>
        <p>Forecast your startup's growth potential and survival chances.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("### 👨‍💻 Created by Aditya Sindhu")
st.caption("Built with Streamlit, Scikit-learn, Pandas, Matplotlib & more.")

