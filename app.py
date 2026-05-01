import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Custom CSS for premium white UI
# -----------------------------
st.markdown("""
<style>
:root {
    --bg: #F5F7FB;
    --card: #FFFFFF;
    --text: #111827;
    --muted: #6B7280;
    --line: #E5E7EB;
    --accent: #22C55E;
    --accent-dark: #16A34A;
    --warning: #F59E0B;
    --danger: #EF4444;
    --shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
    --radius: 20px;
}

.stApp {
    background-color: var(--bg);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #EEF2F7;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--text) !important;
}

.hero-card {
    background: linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1rem 1.2rem;
    box-shadow: var(--shadow);
    min-height: 120px;
}

.metric-label {
    color: var(--muted);
    font-size: 0.92rem;
    font-weight: 600;
}

.metric-value {
    color: var(--text);
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.35rem;
}

.metric-sub {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 0.3rem;
}

.section-heading {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text);
    margin-top: 1.3rem;
    margin-bottom: 0.8rem;
}

.chart-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 1.2rem 1.2rem;
    box-shadow: var(--shadow);
}

.result-pill {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.allow {
    background: #DCFCE7;
    color: #166534;
}

.review {
    background: #FEF3C7;
    color: #92400E;
}

.block {
    background: #FEE2E2;
    color: #991B1B;
}

.result-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text);
}

.result-sub {
    color: var(--muted);
    font-size: 0.95rem;
}

.small-note {
    color: var(--muted);
    font-size: 0.88rem;
}

.stButton > button {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
    color: white;
}

hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def metric_card(title, value, subtext=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_fig(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#111827"),
        title_font=dict(size=18, color="#111827"),
        legend_title_text=""
    )
    fig.update_xaxes(showgrid=False, linecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#EEF2F7", zerolinecolor="#EEF2F7")
    return fig


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
        n_estimators=100,
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
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    rf_model = pipeline.named_steps["classifier"]
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_features = encoder.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(encoded_cat_features)

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(12)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }

    return pipeline, df, metrics, feature_importance_df


model, df, metrics, feature_importance_df = train_model()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## Enter Transaction Details")

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


# -----------------------------
# Derived analytics
# -----------------------------
total_tx = len(df)
fraud_tx = int(df["is_fraud"].sum())
fraud_rate = df["is_fraud"].mean() * 100
fraud_amount = df[df["is_fraud"] == 1]["amount"].sum()

hour_df = df[df["is_fraud"] == 1].groupby("hour").size().reset_index(name="Fraud Transactions")
merchant_df = (
    df.groupby("merchant_category")["is_fraud"]
    .mean()
    .mul(100)
    .reset_index(name="Fraud Percentage")
    .sort_values("Fraud Percentage", ascending=False)
)

channel_df = (
    df.groupby("channel")
    .agg(Total_Transactions=("channel", "size"), Fraud_Rate=("is_fraud", "mean"))
    .reset_index()
)
channel_df["Fraud_Rate"] = channel_df["Fraud_Rate"] * 100

device_df = (
    df.groupby("device_type")["is_fraud"]
    .mean()
    .mul(100)
    .reset_index(name="Fraud Rate")
    .sort_values("Fraud Rate", ascending=False)
)

top_fraud_table = (
    df[df["is_fraud"] == 1]
    .groupby("merchant_category")
    .agg(
        Fraud_Transactions=("is_fraud", "sum"),
        Avg_Amount=("amount", "mean")
    )
    .reset_index()
    .sort_values("Fraud_Transactions", ascending=False)
)


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💳 Credit Card Fraud Detection Dashboard</div>
    <div class="hero-sub">
        Real-time fraud scoring dashboard with machine learning, business KPIs, and transaction risk analytics.
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# KPI Cards
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Total Transactions", f"{total_tx:,}", "Synthetic dataset used for simulation")
with k2:
    metric_card("Fraud Transactions", f"{fraud_tx:,}", "Detected fraudulent records in dataset")
with k3:
    metric_card("Fraud Rate", f"{fraud_rate:.2f}%", "Overall fraud prevalence")
with k4:
    metric_card("Fraud Amount", f"₹{fraud_amount:,.0f}", "Total value of fraudulent transactions")


# -----------------------------
# Charts Row 1
# -----------------------------
st.markdown('<div class="section-heading">Fraud Analytics Overview</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_hour = px.bar(
        hour_df,
        x="hour",
        y="Fraud Transactions",
        title="Fraudulent Transactions by Hour",
        color_discrete_sequence=["#22C55E"]
    )
    style_fig(fig_hour)
    st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_merchant = px.bar(
        merchant_df,
        x="merchant_category",
        y="Fraud Percentage",
        title="Fraud Percentage by Merchant Category",
        color="Fraud Percentage",
        color_continuous_scale="Greens"
    )
    style_fig(fig_merchant)
    st.plotly_chart(fig_merchant, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Charts Row 2
# -----------------------------
c3, c4 = st.columns([1, 1])

with c3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_channel = px.pie(
        channel_df,
        names="channel",
        values="Total_Transactions",
        hole=0.62,
        title="Transaction Share by Channel",
        color_discrete_sequence=["#22C55E", "#CBD5E1"]
    )
    style_fig(fig_channel)
    st.plotly_chart(fig_channel, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig_device = px.bar(
        device_df,
        x="device_type",
        y="Fraud Rate",
        title="Fraud Rate by Device Type",
        color_discrete_sequence=["#16A34A"]
    )
    style_fig(fig_device)
    st.plotly_chart(fig_device, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Live Transaction Scoring
# -----------------------------
st.markdown('<div class="section-heading">Live Transaction Scoring</div>', unsafe_allow_html=True)

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

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

score_col1, score_col2 = st.columns([1.2, 1])

with score_col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("Transaction Preview")
    st.dataframe(transaction, use_container_width=True, hide_index=True)

    if st.button("Predict Fraud Risk"):
        probability = model.predict_proba(transaction)[0][1]
        prediction = model.predict(transaction)[0]

        if probability >= 0.70:
            decision = "BLOCK"
        elif probability >= 0.40:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        st.session_state.prediction_result = {
            "probability": probability,
            "prediction": prediction,
            "decision": decision
        }
    st.markdown('</div>', unsafe_allow_html=True)

with score_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        prob = result["probability"]
        decision = result["decision"]

        if decision == "ALLOW":
            pill_class = "allow"
        elif decision == "REVIEW":
            pill_class = "review"
        else:
            pill_class = "block"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#16A34A"},
                'steps': [
                    {'range': [0, 40], 'color': "#DCFCE7"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ]
            }
        ))
        style_fig(gauge, height=280)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-pill {pill_class}">{decision}</div>
                <div class="result-title">Decision: {decision}</div>
                <div class="result-sub">
                    Fraud probability is <b>{prob:.2%}</b>. 
                    Prediction label: <b>{"Fraud" if result["prediction"] == 1 else "Genuine"}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-title">No prediction yet</div>
                <div class="result-sub">
                    Fill the transaction details in the sidebar and click <b>Predict Fraud Risk</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Model Insights
# -----------------------------
st.markdown('<div class="section-heading">Model Performance & Insights</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    metric_card("Precision", f"{metrics['precision']:.2f}", "How often fraud predictions are correct")
with m2:
    metric_card("Recall", f"{metrics['recall']:.2f}", "How many actual frauds are captured")
with m3:
    metric_card("F1 Score", f"{metrics['f1']:.2f}", "Balance between precision and recall")

ins1, ins2 = st.columns([1, 1])

with ins1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        colorscale=[[0, "#ECFDF5"], [1, "#16A34A"]],
        text=cm,
        texttemplate="%{text}",
        showscale=False
    ))
    cm_fig.update_layout(
        title="Confusion Matrix",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320
    )
    st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with ins2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fi_fig = px.bar(
        feature_importance_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance",
        color_discrete_sequence=["#22C55E"]
    )
    style_fig(fi_fig, height=320)
    st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Summary Table
# -----------------------------
st.markdown('<div class="section-heading">Fraud Summary Table</div>', unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.dataframe(
    top_fraud_table.rename(columns={
        "merchant_category": "Merchant Category",
        "Fraud_Transactions": "Fraud Transactions",
        "Avg_Amount": "Average Fraud Amount"
    }),
    use_container_width=True,
    hide_index=True
)
st.markdown(
    '<div class="small-note">This table highlights the merchant categories with the highest number of fraudulent transactions.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
