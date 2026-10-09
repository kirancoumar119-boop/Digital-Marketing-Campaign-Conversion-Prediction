import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Digital Marketing Campaign Conversion Prediction",
    layout="wide"
)

# 2. Batman: Arkham Knight / WayneTech HUD Styling & Motion FX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap');

    /* Animated Dark Gotham Ambient Background */
    html, body, [class*="css"], .stApp {
        background: linear-gradient(135deg, #03060a 0%, #0a111a 40%, #040810 70%, #020305 100%) !important;
        background-size: 300% 300% !important;
        animation: gothamSky 18s ease infinite !important;
        color: #d1f4ff !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    @keyframes gothamSky {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Batcomputer Scanline HUD Overlay */
    .stApp::before {
        content: " ";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%),
                    linear-gradient(90deg, rgba(0, 229, 255, 0.02), rgba(0, 150, 255, 0.01), rgba(0, 229, 255, 0.02));
        z-index: 1;
        background-size: 100% 4px, 6px 100%;
        pointer-events: none;
        opacity: 0.7;
    }

    /* Cinematic Headers */
    h1, h2, h3, h4, label, .stMarkdown {
        font-family: 'Orbitron', sans-serif !important;
        color: #00e5ff !important;
        letter-spacing: 2px !important;
        text-transform: uppercase;
    }

    h1 {
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.7), 0 0 30px rgba(0, 229, 255, 0.3) !important;
    }

    /* Tactical Dividers */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #005588, #00e5ff, #005588, transparent) !important;
        box-shadow: 0 0 10px #00e5ff !important;
        margin: 2rem 0 !important;
    }

    /* Inputs (Batcomputer Cyber Terminals) */
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {
        background-color: #070d18 !important;
        color: #00e5ff !important;
        border: 1px solid #0088cc !important;
        box-shadow: inset 0 0 8px rgba(0, 229, 255, 0.2) !important;
        border-radius: 0px !important;
        clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px) !important;
    }

    /* Sliders */
    div[data-baseweb="slider"] {
        padding-top: 10px;
    }

    /* WayneTech Tactical HUD Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(6, 14, 26, 0.85) !important;
        border: 1px solid #00e5ff77 !important;
        border-left: 4px solid #00e5ff !important;
        clip-path: polygon(0 0, 94% 0, 100% 16px, 100% 100%, 0 100%) !important;
        padding: 18px !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.15) !important;
        backdrop-filter: blur(8px);
    }

    div[data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.9rem !important;
        text-shadow: 0 0 12px rgba(0, 229, 255, 0.8) !important;
    }

    /* WayneTech Tactical Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #003355 0%, #001a33 100%) !important;
        color: #00e5ff !important;
        border: 1.5px solid #00e5ff !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        clip-path: polygon(14px 0, 100% 0, 100% calc(100% - 14px), calc(100% - 14px) 100%, 0 100%, 0 14px) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4), inset 0 0 10px rgba(0, 229, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        padding: 14px 28px !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        background: #00e5ff !important;
        color: #03060a !important;
        box-shadow: 0 0 35px #00e5ff, inset 0 0 15px #ffffff !important;
        transform: translateY(-2px) !important;
        cursor: pointer !important;
    }

    /* Dataframe Container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #0088cc !important;
        box-shadow: 0 0 15px rgba(0, 136, 204, 0.3) !important;
    }

    /* Top Banner Frame */
    .arkham-banner {
        border: 1px solid #00e5ff;
        border-left: 5px solid #ffab00; /* Tactical Gold Accent */
        padding: 22px;
        background: rgba(8, 16, 30, 0.9);
        clip-path: polygon(0 0, 97% 0, 100% 20px, 100% 100%, 3% 100%, 0 calc(100% - 20px));
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.25), inset 0 0 15px rgba(0, 229, 255, 0.1);
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("marketing_conversion_pipeline.joblib")

try:
    artifacts = load_artifacts()
    preprocessor = artifacts["preprocessor"]
    model = artifacts["model"]
    expected_features = artifacts["feature_names"]
except Exception as e:
    st.error(f"🦇 [WAYNETECH ALERT] // BATCOMPUTER CORE CONNECTION FAILED: {e}")
    st.stop()

# 4. Arkham Knight Banner
st.markdown("""
<div class="arkham-banner">
    <h1 style="margin:0; padding:0;">🦇 WAYNETECH // OS v4.09</h1>
    <p style="margin:6px 0 0 0; color:#00aaff; font-family:'Rajdhani'; font-size:1.1rem;">> BATCOMPUTER THREAT & CONVERSION PREDICTION MODULE // GOTHAM SECTOR</p>
</div>
""", unsafe_allow_html=True)

# 5. Input Controls Setup (3 Tactical Columns)
st.subheader("📡 TARGET DATA MATRIX")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("#### [01] SUBJECT PROFILE")
    gender = st.selectbox("Gender", ["Female", "Male"], key="input_gender")
    age = st.slider("Age", 18, 80, 35, key="input_age")
    income = st.number_input("Income ($)", min_value=10000, max_value=200000, value=55000, step=1000, key="input_income")
    previous_purchases = st.number_input("Previous Transactions", min_value=0, max_value=50, value=1, key="input_purchases")
    loyalty_points = st.number_input("WayneTech Access Index", min_value=0, max_value=10000, value=150, key="input_points")

with col_b:
    st.markdown("#### [02] CAMPAIGN VECTOR")
    campaign_channel = st.selectbox("Campaign Channel", ["PPC", "Referral", "SEO", "Social Media", "Email"], key="input_channel")
    campaign_type = st.selectbox("Campaign Strategy", ["Awareness", "Consideration", "Conversion", "Retention"], key="input_type")
    ad_spend = st.number_input("Ad Budget ($)", min_value=0.0, max_value=10000.0, value=500.0, step=25.0, key="input_adspend")
    ctr = st.slider("Click-Through Rate (CTR)", 0.0, 1.0, 0.05, step=0.001, key="input_ctr")
    conversion_rate = st.slider("Historical Rate", 0.0, 1.0, 0.02, step=0.001, key="input_conv_rate")

with col_c:
    st.markdown("#### [03] SURVEILLANCE LOGS")
    website_visits = st.number_input("Terminal Access Count", min_value=0, max_value=500, value=12, key="input_visits")
    pages_per_visit = st.number_input("Node Depth / Visit", min_value=1.0, max_value=30.0, value=4.5, step=0.5, key="input_pages")
    time_on_site = st.number_input("Session Time (mins)", min_value=0.0, max_value=120.0, value=15.0, step=1.0, key="input_time")
    social_shares = st.number_input("Data Relays", min_value=0, max_value=100, value=2, key="input_shares")
    email_opens = st.number_input("Comm Opens", min_value=0, max_value=100, value=5, key="input_opens")
    email_clicks = st.number_input("Comm Clicks", min_value=0, max_value=50, value=2, key="input_clicks")

# 6. Raw Data & Feature Engineering
raw_input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Income': income,
    'CampaignChannel': campaign_channel,
    'CampaignType': campaign_type,
    'AdSpend': ad_spend,
    'ClickThroughRate': ctr,
    'ConversionRate': conversion_rate,
    'WebsiteVisits': website_visits,
    'PagesPerVisit': pages_per_visit,
    'TimeOnSite': time_on_site,
    'SocialShares': social_shares,
    'EmailOpens': email_opens,
    'EmailClicks': email_clicks,
    'PreviousPurchases': previous_purchases,
    'LoyaltyPoints': loyalty_points
}])

input_df = raw_input_df.copy()
input_df['CTOR'] = input_df['EmailClicks'] / (input_df['EmailOpens'] + 1e-5)
input_df['Engagement_Intensity'] = input_df['PagesPerVisit'] * input_df['TimeOnSite']
input_df['Spend_Per_Visit'] = input_df['AdSpend'] / (input_df['WebsiteVisits'] + 1)
input_df['Loyalty_Velocity'] = input_df['LoyaltyPoints'] / (input_df['PreviousPurchases'] + 1)

# Align columns
input_df = input_df[[col for col in expected_features if col in input_df.columns]]

st.divider()

# 7. Metrics Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🖥️ TELEMETRY STREAM")
    st.dataframe(raw_input_df, hide_index=True)

with col2:
    st.subheader("⚡ DERIVED TACTICAL VECTORS")
    st.metric(label="ENGAGEMENT INTENSITY", value=f"{input_df['Engagement_Intensity'].iloc[0]:.2f}")
    st.metric(label="SPEND PER VISIT", value=f"${input_df['Spend_Per_Visit'].iloc[0]:.2f}")
    st.metric(label="COMM CLICK-TO-OPEN", value=f"{input_df['CTOR'].iloc[0]:.2%}")

st.divider()

# 8. Batcomputer Inference Execution
if st.button("🦇 INITIALIZE BATCOMPUTER INFERENCE", use_container_width=True, key="predict_button"):
    processed_input = preprocessor.transform(input_df)
    conversion_prob = model.predict_proba(processed_input)[0][1]
    prediction = model.predict(processed_input)[0]

    st.subheader("🎯 ANALYSIS RESULT")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(
            label="CONVERSION PROBABILITY", 
            value=f"{conversion_prob * 100:.2f}%"
        )
    
    with res_col2:
        if prediction == 1:
            st.success("⚡ [TARGET SECURED] // HIGH CONVERSION POTENTIAL (CLASS 1)")
        else:
            st.warning("⚠️ [TARGET ELUSIVE] // LOW CONVERSION POTENTIAL (CLASS 0)")

    st.progress(float(conversion_prob))