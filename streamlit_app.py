import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction Model",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

# Default median values for input hints 
DEFAULT_HINTS = {
    'Pregnancies': 3,
    'Glucose': 117,
    'BloodPressure': 72,
    'SkinThickness': 23,
    'Insulin': 30,
    'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.37,
    'Age': 29
}

# Feature definitions
FEATURES_INFO = {
    'Pregnancies': {"label": "Number of Pregnancies", "min": 0, "max": 17, "step": 1, "format": "%d"},
    'Glucose': {"label": "Glucose (mg/dL)", "min": 50, "max": 200, "step": 1, "format": "%d"},
    'BloodPressure': {"label": "Blood Pressure (mm Hg)", "min": 40, "max": 122, "step": 1, "format": "%d"},
    'SkinThickness': {"label": "Skin Thickness (mm)", "min": 5, "max": 99, "step": 1, "format": "%d"},
    'Insulin': {"label": "Insulin (mu U/ml)", "min": 15, "max": 850, "step": 1, "format": "%d"},
    'BMI': {"label": "BMI (kg/m²)", "min": 15.0, "max": 68.0, "step": 0.1, "format": "%.1f"},
    'DiabetesPedigreeFunction': {"label": "Diabetes Pedigree Function", "min": 0.0, "max": 2.5, "step": 0.001, "format": "%.3f"},
    'Age': {"label": "Age (Years)", "min": 21, "max": 81, "step": 1, "format": "%d"}
}

FEATURE_ORDER = list(FEATURES_INFO.keys())

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        with open('optimized_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Failed to load model/scaler: {e}")
        return None, None

model, scaler = load_assets()

# --- UI Styling ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #059669;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #047857;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Diabetes Risk Predictor (SVM Model)")
st.caption("Enter patient diagnostic parameters below to predict diabetes risk.")

if model is None or scaler is None:
    st.stop()

st.header("Patient Input Data")

col1, col2 = st.columns(2)
input_data = {}

# Input widgets
for i, (feature, info) in enumerate(FEATURES_INFO.items()):
    col = col1 if i % 2 == 0 else col2
    with col:
        value = st.number_input(
            info['label'],
            min_value=info['min'],
            max_value=info['max'],
            value=DEFAULT_HINTS.get(feature),
            step=info['step'],
            format=info['format'],
            help=f"Median value: {DEFAULT_HINTS.get(feature)}",
            key=feature
        )
        input_data[feature] = value

# --- Prediction ---
st.markdown("---")

if st.button("Analyze Risk", use_container_width=True):
    try:
        # 1. Prepare input
        input_array = np.array([input_data[f] for f in FEATURE_ORDER])
        input_df = pd.DataFrame([input_array], columns=FEATURE_ORDER)

        # 2. Standardize
        std_data = scaler.transform(input_df)

        # 3. Predict
        prediction = model.predict(std_data)

        # 4. Output
        st.header("Prediction Result")
        if prediction[0] == 0:
            st.success("✅ The model predicts the person is **Not Diabetic** (Low Risk).")
            st.balloons()
        else:
            st.error("⚠️ The model predicts the person is **Diabetic** (High Risk).")
            st.snow()

        # 5. Show model test performance (static, from training notebook)
        st.markdown("**Model Performance on Test Data:**")
        st.markdown("- F1-Score: **0.649**")
        st.markdown("- ROC-AUC: **0.828**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
