import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# Premium White Theme CSS
# -------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        color: #0F172A !important;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .stApp {
        background-color: #F4F6FA !important;
        color: #0F172A !important;
    }

    .main > div {
        padding-top: 1.2rem;
    }

    h1, h2, h3, h4, h5, h6,
    p, span, label, div {
        color: #0F172A;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 246, 250, 0.95);
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #EAF0F7 !important;
        border-right: 1px solid #D9E0EA;
    }

    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
    }

    .hero-box {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 28px 30px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569 !important;
    }

    .metric-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 22px 22px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        border: 1px solid #E5E7EB;
        min-height: 140px;
    }

    .metric-title {
        color: #475569 !important;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #0F172A !important;
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .metric-sub {
        color: #475569 !important;
        font-size: 0.9rem;
    }

    .section-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-top: 26px;
        margin-bottom: 16px;
    }

    .subsection-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 12px;
    }

    .subtle-note {
        color: #475569 !important;
        font-size: 0.92rem;
        margin-top: 12px;
    }

    .result-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 22px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .decision-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .allow-pill {
        background: #DCFCE7;
        color: #166534 !important;
    }

    .review-pill {
        background: #FEF3C7;
        color: #92400E !important;
    }

    .block-pill {
        background: #FEE2E2;
        color: #991B1B !important;
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A !important;
        margin-bottom: 8px;
    }

    .result-text {
        font-size: 1rem;
        color: #475569 !important;
        line-height: 1.6;
    }

    .custom-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }

    .custom-table th {
        background: #F8FAFC;
        color: #0F172A !important;
        text-align: left;
        padding: 14px;
        font-weight: 800;
        border-bottom: 1px solid #E5E7EB;
    }

    .custom-table td {
        padding: 14px;
        border-bottom: 1px solid #EEF2F7;
        color: #0F172A !important;
        font-weight: 500;
    }

    .custom-table tr:last-child td {
        border-bottom: none;
    }

    .stButton > button {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 800;
        width: 100%;
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(34, 197, 94, 0.18);
        color: white !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        overflow: hidden;
        background: #FFFFFF !important;
    }

    .stDataFrame {
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Model Training
# -------------------------------------------------
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
        "f1": f1_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred)
    }

    rf_model = pipeline.named_steps["classifier"]
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_features = encoder.get_feature_names_out(categorical_features)

    all_feature_names = numeric_features + list(encoded_cat_features)

    feature_importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(12)

    return pipeline, df, metrics, feature_importance_df


model, df, metrics, feature_importance_df = train_model()


# -------------------------------------------------
# Plotly Styling Helper
# -------------------------------------------------
def style_fig(fig, height=360):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=13),
        title_font=dict(size=17, color="#0F172A"),
        margin=dict(l=55, r=35, t=60, b=60),
        height=height,
        legend_title_text=""
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13)
    )

    fig.update_yaxes(
        gridcolor="#E5E7EB",
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13)
    )

    fig.update_traces(marker_line_width=0)

    return fig


# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
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


# -------------------------------------------------
# Dashboard Metrics
# -------------------------------------------------
total_tx = len(df)
fraud_tx = int(df["is_fraud"].sum())
fraud_rate = df["is_fraud"].mean() * 100
fraud_amount = df[df["is_fraud"] == 1]["amount"].sum()


# -------------------------------------------------
# Charts Data
# -------------------------------------------------
hour_df = (
    df[df["is_fraud"] == 1]
    .groupby("hour")
    .size()
    .reset_index(name="Fraud Transactions")
)

merchant_df = (
    df.groupby("merchant_category")["is_fraud"]
    .mean()
    .mul(100)
    .reset_index(name="Fraud Percentage")
    .sort_values("Fraud Percentage", ascending=False)
)

channel_df = (
    df.groupby("channel")
    .size()
    .reset_index(name="Transactions")
)

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
        Average_Fraud_Amount=("amount", "mean")
    )
    .reset_index()
    .sort_values("Fraud_Transactions", ascending=False)
)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">💳 Credit Card Fraud Detection Dashboard</div>
    <div class="hero-subtitle">
        Real-time fraud scoring dashboard with machine learning, business KPIs, and transaction risk analytics.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Transactions</div>
        <div class="metric-value">{total_tx:,}</div>
        <div class="metric-sub">Synthetic dataset used for simulation</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fraud Transactions</div>
        <div class="metric-value">{fraud_tx:,}</div>
        <div class="metric-sub">Detected fraudulent records in dataset</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fraud Rate</div>
        <div class="metric-value">{fraud_rate:.2f}%</div>
        <div class="metric-sub">Overall fraud prevalence</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Fraud Amount</div>
        <div class="metric-value">₹{fraud_amount:,.0f}</div>
        <div class="metric-sub">Total value of fraudulent transactions</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# Fraud Analytics Overview
# -------------------------------------------------
st.markdown('<div class="section-title">Fraud Analytics Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_hour = px.bar(
        hour_df,
        x="hour",
        y="Fraud Transactions",
        title="Fraudulent Transactions by Hour",
        color_discrete_sequence=["#22C55E"]
    )
    style_fig(fig_hour)
    st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})

with col2:
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

col3, col4 = st.columns(2)

with col3:
    fig_channel = px.pie(
        channel_df,
        names="channel",
        values="Transactions",
        hole=0.62,
        title="Transaction Share by Channel",
        color_discrete_sequence=["#22C55E", "#CBD5E1"]
    )
    style_fig(fig_channel)
    st.plotly_chart(fig_channel, use_container_width=True, config={"displayModeBar": False})

with col4:
    fig_device = px.bar(
        device_df,
        x="device_type",
        y="Fraud Rate",
        title="Fraud Rate by Device Type",
        color_discrete_sequence=["#16A34A"]
    )
    style_fig(fig_device)
    st.plotly_chart(fig_device, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# Live Transaction Scoring
# -------------------------------------------------
st.markdown('<div class="section-title">Live Transaction Scoring</div>', unsafe_allow_html=True)

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

left, right = st.columns([1.2, 1])

with left:
    st.markdown(
        '<div class="subsection-title">Transaction Preview</div>',
        unsafe_allow_html=True
    )

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

with right:
    if st.session_state.prediction_result is None:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">No prediction yet</div>
            <div class="result-text">
                Fill the transaction details in the sidebar and click <b>Predict Fraud Risk</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state.prediction_result
        prob = result["probability"]
        decision = result["decision"]
        pred_label = "Fraud" if result["prediction"] == 1 else "Genuine"

        if decision == "ALLOW":
            pill_class = "allow-pill"
        elif decision == "REVIEW":
            pill_class = "review-pill"
        else:
            pill_class = "block-pill"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#16A34A"},
                "steps": [
                    {"range": [0, 40], "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70, 100], "color": "#FEE2E2"}
                ]
            }
        ))

        gauge.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            height=360,
            margin=dict(l=35, r=35, t=70, b=30),
            font=dict(color="#0F172A", size=13),
            title_font=dict(color="#0F172A", size=17)
        )

        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div class="result-card">
            <div class="decision-pill {pill_class}">{decision}</div>
            <div class="result-title">Decision: {decision}</div>
            <div class="result-text">
                Fraud probability is <b>{prob:.2%}</b>. Prediction label: <b>{pred_label}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------
# Model Performance
# -------------------------------------------------
st.markdown('<div class="section-title">Model Performance & Insights</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Precision</div>
        <div class="metric-value">{metrics['precision']:.2f}</div>
        <div class="metric-sub">How often fraud predictions are correct</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Recall</div>
        <div class="metric-value">{metrics['recall']:.2f}</div>
        <div class="metric-sub">How many actual frauds are captured</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">F1 Score</div>
        <div class="metric-value">{metrics['f1']:.2f}</div>
        <div class="metric-sub">Balance between precision and recall</div>
    </div>
    """, unsafe_allow_html=True)

a, b = st.columns(2)

with a:
    cm = metrics["cm"]

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
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=13),
        title_font=dict(color="#0F172A", size=17),
        height=360,
        margin=dict(l=55, r=35, t=60, b=60)
    )

    cm_fig.update_xaxes(
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13)
    )

    cm_fig.update_yaxes(
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13)
    )

    st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})

with b:
    fi_fig = px.bar(
        feature_importance_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance",
        color_discrete_sequence=["#22C55E"]
    )

    style_fig(fi_fig)
    st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# Fraud Summary Table
# -------------------------------------------------
st.markdown('<div class="section-title">Fraud Summary Table</div>', unsafe_allow_html=True)

display_table = top_fraud_table.rename(columns={
    "merchant_category": "Merchant Category",
    "Fraud_Transactions": "Fraud Transactions",
    "Average_Fraud_Amount": "Average Fraud Amount"
})

display_table["Average Fraud Amount"] = display_table["Average Fraud Amount"].round(2)

table_html = display_table.to_html(index=False, classes="custom-table")

st.markdown(table_html, unsafe_allow_html=True)

st.markdown(
    '<div class="subtle-note">This table highlights the merchant categories with the highest number of fraudulent transactions.</div>',
    unsafe_allow_html=True
)
