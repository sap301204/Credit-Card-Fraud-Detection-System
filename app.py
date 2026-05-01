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
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 28%),
            linear-gradient(135deg, #F4F7FB 0%, #EDF2F7 100%);
        color: #0F172A;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,0.88);
        backdrop-filter: blur(14px);
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* -----------------------------------------
       SIDEBAR - LIGHT POWER BI / ADMIN STYLE
    ----------------------------------------- */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
        border-right: 1px solid #D8E0EA;
        box-shadow: 10px 0 30px rgba(15,23,42,0.06);
    }

    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    .sidebar-brand {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 16px 16px;
        margin-bottom: 18px;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }

    .sidebar-brand-title {
        font-size: 1.22rem;
        font-weight: 900;
        color: #0F172A !important;
        margin-bottom: 4px;
    }

    .sidebar-brand-sub {
        font-size: 0.84rem;
        color: #64748B !important;
        line-height: 1.45;
    }

    .sidebar-section-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 1.25px;
        color: #64748B !important;
        font-weight: 900;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    /* Sidebar navigation radio */
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 10px 12px;
        margin-bottom: 8px;
        box-shadow: 0 6px 16px rgba(15,23,42,0.04);
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: #F8FAFC;
        border-color: #CBD5E1;
        transform: translateX(2px);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #F8FAFC 0%, #ECFDF5 100%);
        border: 1px solid #86EFAC;
        box-shadow: 0 12px 24px rgba(34,197,94,0.10);
        position: relative;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)::before {
        content: "";
        position: absolute;
        left: 0;
        top: 8px;
        bottom: 8px;
        width: 4px;
        border-radius: 999px;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-weight: 800 !important;
        color: #0F172A !important;
    }

    /* Sidebar inputs */
    [data-testid="stSidebar"] label {
        font-weight: 700 !important;
        color: #334155 !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        box-shadow: 0 4px 12px rgba(15,23,42,0.03);
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] span {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: #EEF2F7 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] span {
        color: #334155 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.78rem 1.15rem !important;
        font-weight: 900 !important;
        box-shadow: 0 12px 26px rgba(34,197,94,0.28);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        color: white !important;
        box-shadow: 0 20px 40px rgba(34,197,94,0.36);
    }

    /* Hero */
    .hero-card {
        background:
            radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 28%),
            linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 28px;
        padding: 30px 34px;
        box-shadow:
            0 18px 42px rgba(15,23,42,0.08),
            0 0 26px rgba(34,197,94,0.05);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 950;
        color: #0F172A;
        margin-bottom: 8px;
        letter-spacing: -0.8px;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        line-height: 1.6;
    }

    /* Section header */
    .section-banner {
        background:
            radial-gradient(circle at top right, rgba(34,197,94,0.08), transparent 28%),
            #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 7px solid #16A34A;
        border-radius: 22px;
        padding: 20px 24px;
        box-shadow:
            0 14px 34px rgba(15,23,42,0.07),
            0 0 18px rgba(34,197,94,0.04);
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 1.9rem;
        font-weight: 950;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.4px;
    }

    .section-subtitle {
        font-size: 0.98rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* KPI Cards */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FCFDFE 100%);
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 24px;
        box-shadow:
            0 18px 36px rgba(15,23,42,0.08),
            0 0 28px rgba(34,197,94,0.04);
        min-height: 150px;
        position: relative;
        overflow: hidden;
        transition: all 0.25s ease;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 5px;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow:
            0 24px 48px rgba(15,23,42,0.12),
            0 0 30px rgba(34,197,94,0.08);
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 850;
        color: #64748B;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 950;
        color: #0F172A;
        margin-bottom: 10px;
    }

    .metric-sub {
        font-size: 0.94rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Container / chart cards pop */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 26px !important;
        box-shadow:
            0 18px 38px rgba(15,23,42,0.08),
            0 0 20px rgba(34,197,94,0.05) !important;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem !important;
        margin-bottom: 24px !important;
        transition: all 0.25s ease !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow:
            0 24px 50px rgba(15,23,42,0.12),
            0 0 28px rgba(34,197,94,0.08) !important;
    }

    .card-title {
        font-size: 1.22rem;
        font-weight: 950;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .card-sub {
        font-size: 0.93rem;
        color: #64748B;
        margin-bottom: 14px;
        line-height: 1.5;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FCFDFE 100%);
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 22px;
        box-shadow:
            0 14px 34px rgba(15,23,42,0.08),
            0 0 18px rgba(34,197,94,0.04);
    }

    .decision-pill {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 0.83rem;
        font-weight: 950;
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
        font-size: 1.7rem;
        font-weight: 950;
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
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 12px;
    }

    .risk-factor-title {
        font-size: 0.95rem;
        font-weight: 950;
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
        border: 1px solid #E2E8F0;
    }

    .custom-table th {
        background: #F8FAFC;
        color: #0F172A;
        text-align: left;
        padding: 13px 15px;
        font-weight: 950;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.9rem;
        white-space: nowrap;
    }

    .custom-table td {
        padding: 13px 15px;
        border-bottom: 1px solid #EEF2F7;
        color: #0F172A;
        font-weight: 650;
        font-size: 0.92rem;
        white-space: nowrap;
    }

    .custom-table tr:last-child td {
        border-bottom: none;
    }

    .preview-table {
        min-width: 1000px;
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


def transaction_signature(tx_df):
    return str(tx_df.to_dict(orient="records")[0])


# -------------------------------------------------
# BUSINESS RISK SCORING
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

    risk += min(amount / 100000, 1.0) * 0.20
    if amount >= 50000:
        factors.append("High transaction amount")

    risk += min(velocity / 150000, 1.0) * 0.24
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
        factors.append("Business-hour transaction")

    if day in [5, 6]:
        risk += 0.04
        factors.append("Weekend transaction")

    if prev_count >= 15:
        risk += 0.10
        factors.append("High previous transaction count")
    elif prev_count >= 8:
        risk += 0.05
        factors.append("Moderate transaction frequency")

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
    else:
        risk -= 0.02

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
    return clamp((0.35 * model_probability) + (0.65 * business_probability), 0.01, 0.99)


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
# SIDEBAR
# -------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">💳 Fraud Ops</div>
    <div class="sidebar-brand-sub">Power BI-style ML fraud analytics dashboard</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard Overview",
        "📊 Fraud Analytics",
        "⚡ Live Scoring",
        "🧠 Model Insights",
        "📋 Fraud Summary"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="sidebar-section-label">Transaction Inputs</div>', unsafe_allow_html=True)

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

current_signature = transaction_signature(transaction)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💳 Credit Card Fraud Detection Dashboard</div>
    <div class="hero-subtitle">
        Premium Power BI-style fraud analytics dashboard with ML scoring, business risk logic, KPI monitoring, and transaction-level decisions.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# COMMON KPI ROW
# -------------------------------------------------
def show_kpis():
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
# SESSION STATE
# -------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_signature" not in st.session_state:
    st.session_state.prediction_signature = None


# -------------------------------------------------
# PAGE: DASHBOARD OVERVIEW
# -------------------------------------------------
if page == "🏠 Dashboard Overview":
    section_header(
        "Dashboard Overview",
        "Executive summary of fraud volume, financial exposure, and model performance."
    )

    show_kpis()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            card_title("Fraudulent Transactions by Hour", "Quick view of risky transaction timings")
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
            card_title("Top Feature Importance", "Most influential variables for fraud decisions")
            fi_fig = px.bar(
                feature_importance_df.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color_discrete_sequence=["#22C55E"]
            )
            style_fig(fi_fig, height=320)
            st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# PAGE: FRAUD ANALYTICS
# -------------------------------------------------
elif page == "📊 Fraud Analytics":
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
            style_fig(fig_hour, height=340)
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
            style_fig(fig_merchant, height=340)
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
                height=340,
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
            style_fig(fig_device, height=340)
            st.plotly_chart(fig_device, use_container_width=True, config={"displayModeBar": False})


# -------------------------------------------------
# PAGE: LIVE SCORING
# -------------------------------------------------
elif page == "⚡ Live Scoring":
    section_header(
        "Live Transaction Scoring",
        "Enter transaction details and click Predict Fraud Risk. The score updates only after clicking the button."
    )

    left, right = st.columns([1.12, 1], gap="large")

    with left:
        with st.container(border=True):
            card_title("Transaction Preview", "Current transaction values from sidebar inputs")
            st.markdown(html_table(preview_df, "preview-table"), unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            if st.button("Predict Fraud Risk", key="predict_button"):
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

                st.session_state.prediction_result = {
                    "probability": probability,
                    "model_probability": model_probability,
                    "business_probability": business_probability,
                    "risk_factors": risk_factors,
                    "decision": decision,
                    "pill_class": pill_class,
                    "pred_label": pred_label
                }

                st.session_state.prediction_signature = current_signature

    with right:
        with st.container(border=True):
            card_title("Fraud Risk Score", "Prediction output for the last scored transaction")

            if st.session_state.prediction_result is None:
                st.markdown("""
                <div class="result-card" style="box-shadow:none; border:none; padding:8px 0;">
                    <div class="result-heading" style="font-size:1.5rem;">No prediction yet</div>
                    <div class="result-text">
                        Fill the transaction details and click <b>Predict Fraud Risk</b> to generate a fraud decision.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif st.session_state.prediction_signature != current_signature:
                st.markdown("""
                <div class="result-card" style="box-shadow:none; border:none; padding:8px 0;">
                    <div class="decision-pill pill-review">INPUTS CHANGED</div>
                    <div class="result-heading" style="font-size:1.5rem;">Click Predict Fraud Risk again</div>
                    <div class="result-text">
                        You changed the transaction inputs. The previous prediction is now outdated.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                result = st.session_state.prediction_result
                probability = result["probability"]
                decision = result["decision"]
                pill_class = result["pill_class"]
                pred_label = result["pred_label"]
                risk_factors = result["risk_factors"]
                model_probability = result["model_probability"]
                business_probability = result["business_probability"]

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
                        "bordercolor": "#E2E8F0",
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
                        Final fraud probability is <b>{probability:.2%}</b>. 
                        Prediction label: <b>{pred_label}</b>.
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
# PAGE: MODEL INSIGHTS
# -------------------------------------------------
elif page == "🧠 Model Insights":
    section_header(
        "Model Performance & Insights",
        "Monitor model quality and understand which features influence fraud decisions."
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
# PAGE: FRAUD SUMMARY
# -------------------------------------------------
elif page == "📋 Fraud Summary":
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
