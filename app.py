import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# ─────────────────────────────────────────────
# 🫀 PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🫀 HeartRisk — CHD Predictor",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 6px 0;
        border-left: 4px solid #e74c3c;
    }
    .risk-low    { border-left-color: #2ecc71 !important; }
    .risk-mid    { border-left-color: #f39c12 !important; }
    .risk-high   { border-left-color: #e74c3c !important; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 📦 LOAD MODEL ASSETS (Synchronized with Notebook)
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    assets = {}
    missing = []
    
    # 1. Load the Model (Matching notebook export name)
    model_file = "heart_disease_model.pkl"
    if os.path.exists(model_file):
        assets["model"] = joblib.load(model_file)
    else:
        missing.append(model_file)
        
    # 2. Load the Scaler (Matching notebook export name)
    scaler_file = "scaler.pkl"
    if os.path.exists(scaler_file):
        assets["scaler"] = joblib.load(scaler_file)
    else:
        missing.append(scaler_file)
        
    # 3. Load Metadata (Threshold & Features)
    config_file = "model_config.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        assets["threshold"] = config.get("optimal_threshold", 0.35)
        assets["features"] = config.get("features", [])
    else:
        # Fallback defaults if json is missing
        assets["threshold"] = 0.35
        assets["features"] = ['age', 'Is_Male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose', 'diaBP', 'BMI', 'PulsePressure']
        
    return assets, missing

assets, missing_files = load_assets()

if missing_files:
    st.error(f"❌ Missing files: {missing_files}. Run the notebook first!")
    st.info(f"📁 Expected filenames: `heart_disease_model.pkl` and `scaler.pkl` in the current folder.")
    st.stop()

model     = assets["model"]
scaler    = assets["scaler"]
threshold = assets["threshold"]
FEATURES  = assets["features"]

FEATURE_LABELS = {
    'age': 'Age',
    'Is_Male': 'Sex (Male=1)',
    'cigsPerDay': 'Cigarettes/Day',
    'totChol': 'Total Cholesterol',
    'sysBP': 'Systolic BP',
    'glucose': 'Glucose',
    'diaBP': 'Diastolic BP',
    'BMI': 'BMI',
    'PulsePressure': 'Pulse Pressure'
}

# ─────────────────────────────────────────────
# 🫀 HEADER
# ─────────────────────────────────────────────
st.title("🫀 HeartRisk — 10-Year CHD Prediction")
st.markdown(
    "Predict your **10-Year Coronary Heart Disease (CHD) risk** using a Logistic Regression "
    "model trained on the **Framingham Heart Study** dataset."
)
st.divider()

# ─────────────────────────────────────────────
# 📝 SIDEBAR — PATIENT INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🩺 Patient Vitals")
    st.caption("Enter patient information below:")

    age     = st.number_input("Age (years)",              min_value=20,  max_value=100, value=45, step=1)
    sex     = st.radio("Sex",                             ["Male", "Female"], horizontal=True)
    cigs    = st.slider("Cigarettes/Day",                 0,  70,   5)
    chol    = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
    sys_bp  = st.slider("Systolic BP (mmHg)",             80, 300, 120)
    dia_bp  = st.slider("Diastolic BP (mmHg)",            40, 160,  80)
    bmi     = st.number_input("BMI (kg/m²)",              min_value=10.0, max_value=60.0, value=26.5, step=0.1, format="%.1f")
    glucose = st.number_input("Glucose (mg/dL)",         min_value=50,  max_value=500, value=90, step=1)

    st.divider()
    st.caption(f"🎯 Decision threshold: **{threshold:.2f}** (auto-tuned via Youden's J)")

# ─────────────────────────────────────────────
# ⚙️ FEATURE ENGINEERING
# ─────────────────────────────────────────────
is_male        = 1 if sex == "Male" else 0
pulse_pressure = sys_bp - dia_bp

# ─────────────────────────────────────────────
# 📊 LIVE METRIC CARDS
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🩸 Systolic BP",     f"{sys_bp} mmHg",
            delta="⚠️ High" if sys_bp > 140 else "✅ Normal",
            delta_color="inverse" if sys_bp > 140 else "normal")
col2.metric("💓 Pulse Pressure",  f"{pulse_pressure} mmHg",
            delta="⚠️ Wide" if pulse_pressure > 60 else "✅ Normal",
            delta_color="inverse" if pulse_pressure > 60 else "normal")
col3.metric("⚖️ BMI",            f"{bmi:.1f}",
            delta="⚠️ Obese" if bmi >= 30 else ("⚠️ Overweight" if bmi >= 25 else "✅ Normal"),
            delta_color="inverse" if bmi >= 25 else "normal")
col4.metric("🚬 Smoking",        f"{cigs} cigs/day",
            delta="⚠️ Active Smoker" if cigs > 0 else "✅ Non-Smoker",
            delta_color="inverse" if cigs > 0 else "normal")

st.divider()

# ─────────────────────────────────────────────
# 🔮 PREDICT BUTTON
# ─────────────────────────────────────────────
left, centre, right = st.columns([1, 2, 1])
with centre:
    predict_btn = st.button("🔮 Generate Diagnostic Report")

if predict_btn:
    # ⚠️ CRITICAL: Ensure this order matches FEATURES list from notebook exactly
    input_data = pd.DataFrame([[age, is_male, cigs, chol, sys_bp, glucose, dia_bp, bmi, pulse_pressure]], 
                              columns=FEATURES)

    try:
        scaled_data = scaler.transform(input_data)
        prob = model.predict_proba(scaled_data)[0][1]
        prediction = 1 if prob >= threshold else 0

        # Risk tiering logic relative to the optimal threshold
        if prob < (threshold / 2):
            tier, tier_color, tier_emoji = "LOW",      "#2ecc71", "🟢"
        elif prob < threshold:
            tier, tier_color, tier_emoji = "MODERATE", "#f39c12", "🟡"
        else:
            tier, tier_color, tier_emoji = "HIGH",     "#e74c3c", "🔴"

        st.divider()
        st.subheader("📋 Diagnostic Report")

        r1, r2 = st.columns(2)
        with r1:
            if prediction == 1:
                st.error(f"### {tier_emoji} {tier} RISK\n**{prob*100:.1f}%** probability of CHD in 10 years")
            else:
                st.success(f"### {tier_emoji} {tier} RISK\n**{prob*100:.1f}%** probability of CHD in 10 years")

        with r2:
            st.markdown("**Risk Probability Gauge:**")
            st.progress(min(prob, 1.0))
            st.caption(f"Calculated using {len(FEATURES)} clinical features.")

    except ValueError as e:
        st.error(f"❌ Shape Mismatch: {e}")
        st.info("The Scaler expects a different number of features.")

# ℹ️ FOOTER
st.divider()
st.caption("🫀 HeartRisk | Dataset: Framingham Heart Study | ⚠️ Educational Use Only")