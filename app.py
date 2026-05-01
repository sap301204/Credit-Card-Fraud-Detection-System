
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("models/fraud_model.pkl")

st.title("💳 Credit Card Fraud Detection System")
st.write("ML-powered fraud detection dashboard using SMOTE and Random Forest.")

st.sidebar.header("Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=75000.0)
hour = st.sidebar.slider("Transaction Hour", 0, 23, 2)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, 6)

merchant_category = st.sidebar.selectbox(
    "Merchant Category",
    ["grocery", "fuel", "restaurant", "shopping", "travel", "electronics", "crypto", "luxury"]
)

device_type = st.sidebar.selectbox(
    "Device Type",
    ["mobile", "desktop", "pos"]
)

channel = st.sidebar.selectbox(
    "Channel",
    ["online", "offline"]
)

is_international = st.sidebar.selectbox(
    "International Transaction?",
    [0, 1]
)

previous_tx_count = st.sidebar.number_input("Previous Transaction Count", min_value=0, value=18)
velocity_amount = st.sidebar.number_input("Velocity Amount", min_value=0.0, value=120000.0)

is_night = 1 if hour < 6 or hour >= 22 else 0

transaction = pd.DataFrame([{
    "amount": amount,
    "hour": hour,
    "day_of_week": day_of_week,
    "merchant_category": merchant_category,
    "device_type": device_type,
    "channel": channel,
    "is_international": is_international,
    "previous_tx_count": previous_tx_count,
    "velocity_amount": velocity_amount,
    "is_night": is_night
}])

st.subheader("Transaction Preview")
st.dataframe(transaction)

if st.button("Predict Fraud Risk"):
    probability = model.predict_proba(transaction)[0][1]
    prediction = model.predict(transaction)[0]

    if probability >= 0.70:
        decision = "BLOCK"
        st.error(f"🚨 High Fraud Risk: {probability:.2%} | Decision: {decision}")
    elif probability >= 0.40:
        decision = "REVIEW"
        st.warning(f"⚠️ Medium Fraud Risk: {probability:.2%} | Decision: {decision}")
    else:
        decision = "ALLOW"
        st.success(f"✅ Low Fraud Risk: {probability:.2%} | Decision: {decision}")

st.markdown("---")

st.subheader("Decision Logic")
st.write("""
- Fraud Probability below 40% → ALLOW  
- Fraud Probability between 40% and 70% → REVIEW  
- Fraud Probability above 70% → BLOCK  
""")

st.subheader("Project Summary")
st.write("""
This dashboard simulates a banking fraud detection system where transaction details are entered,
and a machine learning model predicts whether the transaction should be allowed, reviewed, or blocked.
""")
