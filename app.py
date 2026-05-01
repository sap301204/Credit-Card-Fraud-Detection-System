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
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# CSS - POWER BI STYLE PREMIUM UI
# -------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #F4F7FB;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.95);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #EAF1F8;
        border-right: 1px solid #D7E0EA;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #0F172A !important;
    }

    [data-testid="stNumberInput"] input {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
    }

    [data-testid="stNumberInput"] button {
        color: #0F172A !important;
    }

    div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        min-height: 44px !important;
    }

    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }

    [data-baseweb="popover"] * {
        color: #0F172A !important;
    }

    [data-testid="stSlider"] label,
    [data-testid="stSlider"] span {
        color: #0F172A !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.78rem 1.15rem !important;
        font-weight: 900 !important;
        box-shadow: 0 12px 24px rgba(34, 197, 94, 0.22);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        color: white !important;
        box-shadow: 0 18px 36px rgba(34, 197, 94, 0.30);
    }

    /* Hero */
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E5E7EB;
        border-radius: 28px;
        padding: 32px 36px;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 8px;
        height: 100%;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 8px;
        letter-spacing: -0.7px;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
    }

    /* Section header */
    .section-banner {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 7px solid #16A34A;
        border-radius: 22px;
        padding: 20px 24px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        margin-top: 30px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 1.85rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 5px;
        letter-spacing: -0.4px;
    }

    .section-subtitle {
        font-size: 0.98rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* KPI Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 24px;
        padding: 24px 24px;
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.09);
        min-height: 155px;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 6px;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 22px 48px rgba(15, 23, 42, 0.14);
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 800;
        color: #64748B;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2.25rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 10px;
    }

    .metric-sub {
        font-size: 0.94rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Power BI-like chart cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 26px !important;
        box-shadow:
            0 16px 38px rgba(15, 23, 42, 0.09),
            0 4px 10px rgba(15, 23, 42, 0.04) !important;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem !important;
        margin-bottom: 24px !important;
        transition: all 0.25s ease-in-out !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 22px 48px rgba(15, 23, 42, 0.13),
            0 8px 18px rgba(15, 23, 42, 0.06) !important;
    }

    .card-title {
        font-size: 1.22rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .card-sub {
        font-size: 0.93rem;
        color: #64748B;
        margin-bottom: 14px;
        line-height: 1.5;
    }

    /* Result cards */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    }

    .decision-pill {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 900;
        margin-bottom: 14px;
    }

    .pill-allow {
        background: #DCFCE7;
        color: #166534;
    }

    .pill-review {
        background: #FEF3C7;
        color: #92400E;
    }

    .pill-block {
        background: #FEE2E2;
        color: #991B1B;
    }

    .result-heading {
        font-size: 1.75rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 1rem;
        color: #475569;
        line-height: 1.6;
    }

    .risk-factor-box {
        background: #F8FAFC;
        border: 1px solid #E5E7EB;
        border-radius: 18px;
        padding: 14px 16px;
        margin-top: 12px;
    }

    .risk-factor-title {
        font-size: 0.95rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .risk-factor-text {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.6;
    }

    .note-text {
        color: #475569;
        font-size: 0.94rem;
        margin-top: 12px;
    }

    /* Tables */
    .table-scroll {
        width: 100%;
        overflow-x: auto;
        border-radius: 18px;
    }

    .custom-table {
        width: 100%;
        min-width: 820px;
        border-collapse: collapse;
        background: #FFFFFF;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #E5E7EB;
    }

    .custom-table th {
        background: #F8FAFC;
        color: #0F172A;
        text-align: left;
        padding: 13px 15px;
        font-weight: 900;
        border-bottom: 1px solid #E5E7EB;
        font-size: 0.9rem;
        white-space: nowrap;
    }

    .custom-table td {
        padding: 13px 15px;
        border-bottom: 1px solid #EEF2F7;
        color: #0F172A;
        font-weight: 600;
        font-size: 0.92rem;
        white-space: nowrap;
    }

    .custom-table tr:last-child td {
        border-bottom: none;
    }

    .preview-table {
        min-width: 980px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def render_metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def card_title(title, subtitle=None):
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="card-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", size=13),
        margin=dict(l=55, r=25, t=10, b=55),
        height=height,
        showlegend=False
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13),
        automargin=True
    )
    fig.update_yaxes(
        gridcolor="#E5E7EB",
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=12),
        title_font=dict(color="#334155", size=13),
        automargin=True
    )
    return fig


def html_table(df, extra_class=""):
    return (
        f'<div class="table-scroll">'
        f'{df.to_html(index=False, classes=f"custom-table {extra_class}", escape=False)}'
        f'</div>'
    )


def clamp(value, low=0.01, high=0.99):
    return max(low, min(high, value))


# -------------------------------------------------
# BUSINESS RISK SCORING LAYER
# -------------------------------------------------
def calculate_business_risk(tx):
    amount = float(tx["amount"])
    hour = int(tx["hour"])
    day = int(tx["day_of_week"])
    merchant = str(tx["merchant_category"]).lower()
    device = str(tx["device_type"]).lower()
    channel = str(tx["channel"]).lower()
    international = int(tx["is_international"])
    prev_count = int(tx["previous_tx_count"])
    velocity = float(tx["velocity_amount"])
    night = int(tx["is_night"])

    risk = 0.03
    factors = []

    amount_score = min(amount / 100000, 1.0) * 0.20
    risk += amount_score
    if amount >= 50000:
        factors.append("High transaction amount")

    velocity_score = min(velocity / 150000, 1.0) * 0.24
    risk += velocity_score
    if velocity >= 70000:
        factors.append("High transaction velocity")

    if international == 1:
        risk += 0.14
        factors.append("International transaction")

    if night == 1:
        risk += 0.12
        factors.append("Night-time transaction")

    if hour in [0, 1, 2, 3, 4]:
        risk += 0.08
        factors.append("High-risk transaction hour")
    elif hour in [22, 23]:
        risk += 0.05
        factors.append("Late-hour transaction")
    elif 9 <= hour <= 18:
        risk -= 0.04

    if day in [5, 6]:
        risk += 0.04
        factors.append("Weekend transaction")

    if prev_count >= 15:
        risk += 0.10
        factors.append("High previous transaction count")
    elif prev_count >= 8:
        risk += 0.05

    merchant_risk = {
        "crypto": 0.20,
        "luxury": 0.18,
        "electronics": 0.12,
        "travel": 0.08,
        "shopping": 0.05,
        "restaurant": 0.02,
        "fuel": 0.01,
        "grocery": -0.03
    }
    risk += merchant_risk.get(merchant, 0.03)

    if merchant in ["crypto", "luxury", "electronics", "travel"]:
        factors.append(f"Risky merchant category: {merchant.title()}")

    if channel == "online":
        risk += 0.05
        factors.append("Online transaction channel")

    if device == "mobile":
        risk += 0.03
    elif device == "desktop":
        risk += 0.02
    elif device == "pos":
        risk -= 0.03

    risk = clamp(risk, 0.01, 0.98)

    if not factors:
        factors.append("Low-risk behavioral pattern")

    return risk, factors


def final_fraud_probability(model_probability, business_probability):
    final_probability = (0.35 * model_probability) + (0.65 * business_probability)
    return clamp(final_probability, 0.01, 0.99)


# -------------------------------------------------
# MODEL TRAINING
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

    feature_names = numeric_features + list(encoded_cat_features)

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(12)

    feature_importance_df["Feature"] = (
        feature_importance_df["Feature"]
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    return pipeline, df, metrics, feature_importance_df


model, df, metrics, feature_importance_df = train_model()


# -------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------
st.sidebar.markdown("## Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=75000.0, step=1000.0)
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
velocity_amount = st.sidebar.number_input("Velocity Amount", min_value=0.0, value=120000.0, step=1000.0)

is_night = 1 if hour < 6 or hour >= 22 else 0


# -------------------------------------------------
# DERIVED DATA
# -------------------------------------------------
total_tx = len(df)
fraud_tx = int(df["is_fraud"].sum())
fraud_rate = df["is_fraud"].mean() * 100
fraud_amount = df[df["is_fraud"] == 1]["amount"].sum()

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
merchant_df["merchant_category"] = merchant_df["merchant_category"].str.title()

channel_df = (
    df.groupby("channel")
    .size()
    .reset_index(name="Transactions")
)
channel_df["channel"] = channel_df["channel"].str.title()

device_df = (
    df.groupby("device_type")["is_fraud"]
    .mean()
    .mul(100)
    .reset_index(name="Fraud Rate")
    .sort_values("Fraud Rate", ascending=False)
)
device_df["device_type"] = device_df["device_type"].str.title()

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

top_fraud_table["merchant_category"] = top_fraud_table["merchant_category"].str.title()
top_fraud_table["Average_Fraud_Amount"] = top_fraud_table["Average_Fraud_Amount"].round(2)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💳 Credit Card Fraud Detection Dashboard</div>
    <div class="hero-subtitle">
        Premium fraud analytics dashboard with ML scoring, business risk rules, KPI monitoring, and transaction-level fraud decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4, gap="large")

with k1:
    render_metric_card("Total Transactions", f"{total_tx:,}", "Synthetic dataset used for simulation")

with k2:
    render_metric_card("Fraud Transactions", f"{fraud_tx:,}", "Detected fraudulent records in dataset")

with k3:
    render_metric_card("Fraud Rate", f"{fraud_rate:.2f}%", "Overall fraud prevalence")

with k4:
    render_metric_card("Fraud Amount", f"₹{fraud_amount:,.0f}", "Total value of fraudulent transactions")


# -------------------------------------------------
# SECTION 1: FRAUD ANALYTICS
# -------------------------------------------------
section_header(
    "Fraud Analytics Overview",
    "Explore fraud behavior across time, merchant category, channel, and device type."
)

c1, c2 = st.columns(2, gap="large")

with c1:
    with st.container(border=True):
        card_title("Fraudulent Transactions by Hour", "Hourly pattern of detected fraud cases")
        fig_hour = px.bar(
            hour_df,
            x="hour",
            y="Fraud Transactions",
            labels={"hour": "Hour", "Fraud Transactions": "Fraud Transactions"},
            color_discrete_sequence=["#22C55E"]
        )
        style_fig(fig_hour, height=320)
        st.plotly_chart(fig_hour, use_container_width=True, config={"displayModeBar": False})

with c2:
    with st.container(border=True):
        card_title("Fraud Percentage by Merchant Category", "Merchant categories ranked by fraud share")
        fig_merchant = px.bar(
            merchant_df,
            x="merchant_category",
            y="Fraud Percentage",
            labels={
                "merchant_category": "Merchant Category",
                "Fraud Percentage": "Fraud Percentage (%)"
            },
            color_discrete_sequence=["#16A34A"]
        )
        style_fig(fig_merchant, height=320)
        fig_merchant.update_xaxes(tickangle=30)
        st.plotly_chart(fig_merchant, use_container_width=True, config={"displayModeBar": False})

c3, c4 = st.columns(2, gap="large")

with c3:
    with st.container(border=True):
        card_title("Transaction Share by Channel", "Distribution of transactions across channels")
        fig_channel = px.pie(
            channel_df,
            names="channel",
            values="Transactions",
            hole=0.60,
            color_discrete_sequence=["#22C55E", "#CBD5E1"]
        )
        fig_channel.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#0F172A", size=13),
            height=320,
            margin=dict(l=25, r=25, t=10, b=10),
            legend_title_text="",
            showlegend=True
        )
        st.plotly_chart(fig_channel, use_container_width=True, config={"displayModeBar": False})

with c4:
    with st.container(border=True):
        card_title("Fraud Rate by Device Type", "Fraud risk split by device source")
        fig_device = px.bar(
            device_df,
            x="device_type",
            y="Fraud Rate",
            labels={"device_type": "Device Type", "Fraud Rate": "Fraud Rate (%)"},
            color_discrete_sequence=["#16A34A"]
        )
        style_fig(fig_device, height=320)
        st.plotly_chart(fig_device, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# SECTION 2: LIVE SCORING
# -------------------------------------------------
section_header(
    "Live Transaction Scoring",
    "Enter transaction details and generate a dynamic fraud score. The score updates using ML probability plus business-risk logic."
)

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

preview_df = transaction.copy()
preview_df.columns = [
    "Amount",
    "Hour",
    "Day Of Week",
    "Merchant Category",
    "Device Type",
    "Channel",
    "International",
    "Previous Tx Count",
    "Velocity Amount",
    "Night Tx"
]

preview_df["Amount"] = preview_df["Amount"].map(lambda x: f"₹{x:,.2f}")
preview_df["Velocity Amount"] = preview_df["Velocity Amount"].map(lambda x: f"₹{x:,.2f}")
preview_df["Merchant Category"] = preview_df["Merchant Category"].str.title()
preview_df["Device Type"] = preview_df["Device Type"].str.title()
preview_df["Channel"] = preview_df["Channel"].str.title()
preview_df["International"] = preview_df["International"].map({1: "Yes", 0: "No"})
preview_df["Night Tx"] = preview_df["Night Tx"].map({1: "Yes", 0: "No"})

if "has_scored" not in st.session_state:
    st.session_state.has_scored = False

left, right = st.columns([1.12, 1], gap="large")

with left:
    with st.container(border=True):
        card_title("Transaction Preview", "Transaction values to be scored")
        st.markdown(html_table(preview_df, "preview-table"), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("Predict Fraud Risk", key="predict_button"):
            st.session_state.has_scored = True

with right:
    with st.container(border=True):
        card_title("Fraud Risk Score", "Prediction output for current transaction")

        if not st.session_state.has_scored:
            st.markdown("""
            <div class="result-card" style="box-shadow:none; border:none; padding:8px 0;">
                <div class="result-heading" style="font-size:1.5rem;">No prediction yet</div>
                <div class="result-text">
                    Fill the transaction details and click <b>Predict Fraud Risk</b>. After the first click, the score will update automatically when inputs change.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            model_probability = float(model.predict_proba(transaction)[0][1])
            business_probability, risk_factors = calculate_business_risk(transaction.iloc[0])
            probability = final_fraud_probability(model_probability, business_probability)

            if probability >= 0.70:
                decision = "BLOCK"
                pill_class = "pill-block"
                pred_label = "Fraud"
            elif probability >= 0.40:
                decision = "REVIEW"
                pill_class = "pill-review"
                pred_label = "Fraud"
            else:
                decision = "ALLOW"
                pill_class = "pill-allow"
                pred_label = "Genuine"

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%",
                    "font": {"size": 58, "color": "#0F172A"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#475569"
                    },
                    "bar": {"color": "#16A34A", "thickness": 0.34},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#E5E7EB",
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
                font=dict(color="#0F172A", size=13),
                margin=dict(l=25, r=25, t=10, b=10),
                height=280
            )

            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
            <div class="result-card" style="margin-top:10px;">
                <div class="decision-pill {pill_class}">{decision}</div>
                <div class="result-heading">Decision: {decision}</div>
                <div class="result-text">
                    Final fraud probability is <b>{probability:.2%}</b>. Prediction label: <b>{pred_label}</b>.
                </div>
                <div class="risk-factor-box">
                    <div class="risk-factor-title">Top Risk Signals</div>
                    <div class="risk-factor-text">{", ".join(risk_factors[:5])}</div>
                </div>
                <div class="risk-factor-box">
                    <div class="risk-factor-title">Score Breakdown</div>
                    <div class="risk-factor-text">
                        ML probability: <b>{model_probability:.2%}</b><br>
                        Business-risk probability: <b>{business_probability:.2%}</b><br>
                        Final blended score: <b>{probability:.2%}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------
# SECTION 3: MODEL PERFORMANCE
# -------------------------------------------------
section_header(
    "Model Performance & Insights",
    "Monitor model quality and understand which transaction features influence fraud decisions."
)

m1, m2, m3 = st.columns(3, gap="large")

with m1:
    render_metric_card("Precision", f"{metrics['precision']:.2f}", "How often fraud predictions are correct")

with m2:
    render_metric_card("Recall", f"{metrics['recall']:.2f}", "How many actual frauds are captured")

with m3:
    render_metric_card("F1 Score", f"{metrics['f1']:.2f}", "Balance between precision and recall")

a, b = st.columns(2, gap="large")

with a:
    with st.container(border=True):
        card_title("Confusion Matrix", "Actual vs predicted classification results")

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
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#0F172A", size=13),
            margin=dict(l=65, r=25, t=10, b=55),
            height=340
        )

        cm_fig.update_xaxes(
            tickfont=dict(color="#334155", size=12),
            title_font=dict(color="#334155", size=13),
            automargin=True
        )

        cm_fig.update_yaxes(
            tickfont=dict(color="#334155", size=12),
            title_font=dict(color="#334155", size=13),
            automargin=True
        )

        st.plotly_chart(cm_fig, use_container_width=True, config={"displayModeBar": False})

with b:
    with st.container(border=True):
        card_title("Top Feature Importance", "Most influential variables for model decisions")

        fi_fig = px.bar(
            feature_importance_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#22C55E"]
        )

        style_fig(fi_fig, height=340)
        st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# SECTION 4: FRAUD SUMMARY
# -------------------------------------------------
section_header(
    "Fraud Summary Table",
    "Merchant-level fraud summary for quick fraud operations review."
)

display_table = top_fraud_table.rename(columns={
    "merchant_category": "Merchant Category",
    "Fraud_Transactions": "Fraud Transactions",
    "Average_Fraud_Amount": "Average Fraud Amount"
}).copy()

display_table["Average Fraud Amount"] = display_table["Average Fraud Amount"].map(lambda x: f"₹{x:,.2f}")

with st.container(border=True):
    st.markdown(html_table(display_table), unsafe_allow_html=True)
    st.markdown(
        '<div class="note-text">This table highlights merchant categories with the highest number of fraudulent transactions.</div>',
        unsafe_allow_html=True
    )
