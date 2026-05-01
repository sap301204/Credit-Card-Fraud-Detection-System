import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def train_model():
    df = pd.read_csv("transactions.csv")

    X = df.drop(columns=["is_fraud", "transaction_id"])
    y = df["is_fraud"]

    numeric_features = [
        "amount",
        "hour",
        "day_of_week",
        "is_international",
        "previous_tx_count",
        "velocity_amount",
        "is_night"
    ]

    categorical_features = [
        "merchant_category",
        "device_type",
        "channel"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return pipeline, df, metrics


model, df, metrics = train_model()


st.title("💳 Credit Card Fraud Detection System")
st.write(
    "This machine learning dashboard predicts whether a credit card transaction is genuine or fraudulent."
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    st.metric("Fraud Transactions", int(df["is_fraud"].sum()))

with col3:
    fraud_rate = df["is_fraud"].mean() * 100
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

with col4:
    st.metric("Model Used", "Random Forest")


st.markdown("---")

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

previous_tx_count = st.sidebar.number_input(
    "Previous Transaction Count",
    min_value=0,
    value=18
)

velocity_amount = st.sidebar.number_input(
    "Velocity Amount",
    min_value=0.0,
    value=120000.0
)

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


st.subheader("🔍 Transaction Preview")
st.dataframe(transaction, use_container_width=True)


if st.button("Predict Fraud Risk"):
    probability = model.predict_proba(transaction)[0][1]
    prediction = model.predict(transaction)[0]

    st.subheader("Prediction Result")

    if probability >= 0.70:
        decision = "BLOCK"
        st.error(f"🚨 High Fraud Risk: {probability:.2%} | Decision: {decision}")

    elif probability >= 0.40:
        decision = "REVIEW"
        st.warning(f"⚠️ Medium Fraud Risk: {probability:.2%} | Decision: {decision}")

    else:
        decision = "ALLOW"
        st.success(f"✅ Low Fraud Risk: {probability:.2%} | Decision: {decision}")

    result_df = pd.DataFrame([{
        "Fraud Probability": f"{probability:.2%}",
        "Prediction": "Fraud" if prediction == 1 else "Genuine",
        "Decision": decision
    }])

    st.dataframe(result_df, use_container_width=True)


st.markdown("---")

st.subheader("📊 Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Precision", f"{metrics['precision']:.2f}")

with metric_col2:
    st.metric("Recall", f"{metrics['recall']:.2f}")

with metric_col3:
    st.metric("F1 Score", f"{metrics['f1_score']:.2f}")


st.subheader("Confusion Matrix")

cm = metrics["confusion_matrix"]

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted Genuine", "Predicted Fraud"],
    index=["Actual Genuine", "Actual Fraud"]
)

st.dataframe(cm_df, use_container_width=True)


st.markdown("---")

st.subheader("Decision Logic")

st.write("""
| Fraud Probability | Decision |
|---|---|
| Below 40% | ALLOW |
| 40% to 70% | REVIEW |
| Above 70% | BLOCK |
""")


st.subheader("Project Summary")

st.write("""
This project simulates a banking fraud detection system.
It uses synthetic transaction data, SMOTE imbalance handling, Random Forest classification,
and a Streamlit dashboard to classify transactions into ALLOW, REVIEW, or BLOCK.
""")
