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
# CSS - PREMIUM POWER BI STYLE
# -------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #F5F7FB;
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.95);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #EEF3F8;
        border-right: 1px solid #DCE4EE;
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

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.1rem !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 18px rgba(34, 197, 94, 0.18);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        color: white !important;
        box-shadow: 0 12px 25px rgba(34, 197, 94, 0.26);
    }

    /* Header */
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 26px;
        padding: 30px 34px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 2.35rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        color: #475569;
        line-height: 1.6;
    }

    /* Section */
    .section-shell {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        margin-top: 24px;
        margin-bottom: 26px;
    }

    .section-title {
        font-size: 1.85rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }

    .section-subtitle {
        font-size: 0.98rem;
        color: #64748B;
        margin-bottom: 20px;
    }

    /* KPI Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 24px;
        padding: 22px 24px;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        min-height: 150px;
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
        width: 5px;
        background: linear-gradient(180deg, #22C55E, #16A34A);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.13);
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 800;
        color: #64748B;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 10px;
    }

    .metric-sub {
        font-size: 0.94rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Chart / card containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 24px !important;
        box-shadow:
            0 14px 34px rgba(15, 23, 42, 0.08),
            0 4px 10px rgba(15, 23, 42, 0.04) !important;
        padding: 1.1rem 1.1rem 1.2rem 1.1rem !important;
        margin-bottom: 22px !important;
        transition: all 0.25s ease-in-out !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow:
            0 20px 45px rgba(15, 23, 42, 0.12),
            0 8px 16px rgba(15, 23, 42, 0.05) !important;
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
    }

    /* Result */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
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

    .small-muted {
        color: #64748B;
        font-size: 0.9rem;
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
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
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
        Real-time fraud scoring dashboard with machine learning, business KPIs, and transaction risk analytics.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

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
    "Explore fraud behavior across time, merchant categories, channels, and devices."
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
    "Score a transaction instantly and classify it as ALLOW, REVIEW, or BLOCK."
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

# Unique input signature to avoid stale prediction
current_signature = str(transaction.to_dict(orient="records")[0])

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_signature" not in st.session_state:
    st.session_state.prediction_signature = None

left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.container(border=True):
        card_title("Transaction Preview", "Transaction values to be scored")
        st.markdown(html_table(preview_df, "preview-table"), unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        if st.button("Predict Fraud Risk", key="predict_button"):
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

            st.session_state.prediction_signature = current_signature

with right:
    with st.container(border=True):
        card_title("Fraud Risk Score", "Prediction output for current transaction")

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
                <div class="result-heading" style="font-size:1.5rem;">Run prediction again</div>
                <div class="result-text">
                    Transaction values have changed. Click <b>Predict Fraud Risk</b> again to update the score.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            result = st.session_state.prediction_result
            prob = result["probability"]
            decision = result["decision"]
            pred_label = "Fraud" if result["prediction"] == 1 else "Genuine"

            if decision == "ALLOW":
                pill_class = "pill-allow"
            elif decision == "REVIEW":
                pill_class = "pill-review"
            else:
                pill_class = "pill-block"

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
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
                    Fraud probability is <b>{prob:.2%}</b>. Prediction label: <b>{pred_label}</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------
# SECTION 3: MODEL PERFORMANCE
# -------------------------------------------------
section_header(
    "Model Performance & Insights",
    "Monitor classification quality and identify the most important fraud-driving features."
)

m1, m2, m3 = st.columns(3)

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
# SECTION 4: FRAUD SUMMARY TABLE
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
        '<div class="note-text">This table highlights the merchant categories with the highest number of fraudulent transactions.</div>',
        unsafe_allow_html=True
    )
