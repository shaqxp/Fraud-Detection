import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Fraud Detection System", page_icon="🔐", layout="wide")

MODEL_PATH = Path("fraud_model.pkl")
SCALER_PATH = Path("scaler.pkl")

THRESHOLD = 0.40

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

st.title("🔐 Transaction Fraud Detection System")
st.write("Enter transaction details below to predict whether a transaction is fraudulent.")

if model is None or scaler is None:
    st.error("Model files not found. Make sure `fraud_model.pkl` and `scaler.pkl` are in the same folder as this app.")
    st.stop()

feature_names = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

default_values = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
}

st.sidebar.header("About")
st.sidebar.write(
    "This app predicts fraud probability using a trained machine learning model."
)
st.sidebar.write(f"Decision threshold: **{THRESHOLD}**")

st.subheader("Input Transaction Details")

with st.form("fraud_form"):
    cols = st.columns(3)
    inputs = {}

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                feature,
                value=float(default_values[feature]),
                format="%.6f"
            )

    submitted = st.form_submit_button("Check Transaction")

if submitted:
    input_df = pd.DataFrame([inputs], columns=feature_names)

    input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= THRESHOLD)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Safe Transaction")

    st.metric("Fraud Probability", f"{probability:.4f}")

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.70:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.write(f"**Risk Level:** {risk}")

    st.subheader("Entered Transaction Data")
    st.dataframe(pd.DataFrame([inputs]), use_container_width=True)