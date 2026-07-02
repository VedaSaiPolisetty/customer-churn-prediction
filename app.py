"""
Customer Churn Prediction App
Veda Sai Polisetty | Data Analyst + AI/ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Churn Predictor | Veda Sai Polisetty",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
  .metric-card {
    background: #f0f4ff;
    border-left: 4px solid #1a56db;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
  }
  .risk-high   { background:#fff0f0; border-left:4px solid #e74c3c; border-radius:8px; padding:1rem 1.25rem; }
  .risk-medium { background:#fffbf0; border-left:4px solid #f39c12; border-radius:8px; padding:1rem 1.25rem; }
  .risk-low    { background:#f0fff4; border-left:4px solid #10b981; border-radius:8px; padding:1rem 1.25rem; }
  .section-header { font-size:1.1rem; font-weight:600; color:#1e293b; margin:1.5rem 0 0.5rem; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("outputs/churn_model.pkl", "rb") as f:
        return pickle.load(f)

payload  = load_model()
model    = payload["model"]
features = payload["features"]

# ── Header ──────────────────────────────────────────────────────────────────

st.title("📊 Customer Churn Predictor")
st.markdown(
    "**Built by Veda Sai Polisetty** · Trained on 7,032 IBM Telco customers · "
    "AUC: **0.8346** · [GitHub](https://github.com) · [LinkedIn](https://linkedin.com)"
)
st.divider()

# ── Layout ──────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown("### Customer profile")
    st.caption("Fill in the customer details to get a churn probability score.")

    gender         = st.selectbox("Gender",           ["Male", "Female"])
    senior         = st.selectbox("Senior citizen",   ["No", "Yes"])
    partner        = st.selectbox("Has partner",      ["Yes", "No"])
    dependents     = st.selectbox("Has dependents",   ["No", "Yes"])
    tenure         = st.slider("Tenure (months)",     0, 72, 12)
    phone          = st.selectbox("Phone service",    ["Yes", "No"])
    multi_lines    = st.selectbox("Multiple lines",   ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    online_sec     = st.selectbox("Online security",  ["No", "Yes", "No internet service"])
    online_backup  = st.selectbox("Online backup",    ["Yes", "No", "No internet service"])
    device_prot    = st.selectbox("Device protection",["No", "Yes", "No internet service"])
    tech_support   = st.selectbox("Tech support",     ["No", "Yes", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
    streaming_mov  = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    contract       = st.selectbox("Contract type",    ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless billing",["Yes", "No"])
    payment        = st.selectbox("Payment method",   [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly        = st.number_input("Monthly charges ($)", 18.0, 120.0, 65.0, step=1.0)
    total          = st.number_input("Total charges ($)",   18.0, 9000.0, monthly * tenure, step=10.0)

    predict_btn = st.button("Predict churn risk ↗", type="primary", use_container_width=True)

with right:
    if predict_btn:
        le_map = {
            "gender":          {"Male": 1, "Female": 0},
            "Partner":         {"Yes": 1, "No": 0},
            "Dependents":      {"Yes": 1, "No": 0},
            "PhoneService":    {"Yes": 1, "No": 0},
            "MultipleLines":   {"No": 0, "Yes": 1, "No phone service": 2},
            "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
            "OnlineSecurity":  {"No": 0, "Yes": 1, "No internet service": 2},
            "OnlineBackup":    {"No": 0, "Yes": 1, "No internet service": 2},
            "DeviceProtection":{"No": 0, "Yes": 1, "No internet service": 2},
            "TechSupport":     {"No": 0, "Yes": 1, "No internet service": 2},
            "StreamingTV":     {"No": 0, "Yes": 1, "No internet service": 2},
            "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
            "Contract":        {"Month-to-month": 0, "One year": 1, "Two year": 2},
            "PaperlessBilling":{"Yes": 1, "No": 0},
            "PaymentMethod":   {
                "Bank transfer (automatic)": 0,
                "Credit card (automatic)": 1,
                "Electronic check": 2,
                "Mailed check": 3,
            },
        }

        row = {
            "gender":          le_map["gender"][gender],
            "SeniorCitizen":   1 if senior == "Yes" else 0,
            "Partner":         le_map["Partner"][partner],
            "Dependents":      le_map["Dependents"][dependents],
            "tenure":          tenure,
            "PhoneService":    le_map["PhoneService"][phone],
            "MultipleLines":   le_map["MultipleLines"][multi_lines],
            "InternetService": le_map["InternetService"][internet],
            "OnlineSecurity":  le_map["OnlineSecurity"][online_sec],
            "OnlineBackup":    le_map["OnlineBackup"][online_backup],
            "DeviceProtection":le_map["DeviceProtection"][device_prot],
            "TechSupport":     le_map["TechSupport"][tech_support],
            "StreamingTV":     le_map["StreamingTV"][streaming_tv],
            "StreamingMovies": le_map["StreamingMovies"][streaming_mov],
            "Contract":        le_map["Contract"][contract],
            "PaperlessBilling":le_map["PaperlessBilling"][paperless],
            "PaymentMethod":   le_map["PaymentMethod"][payment],
            "MonthlyCharges":  monthly,
            "TotalCharges":    total,
        }

        X_input = pd.DataFrame([row])[features]
        prob    = model.predict_proba(X_input)[0][1]
        pct     = prob * 100

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = round(pct, 1),
            number= {"suffix": "%", "font": {"size": 40}},
            title = {"text": "Churn probability", "font": {"size": 16}},
            gauge = {
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#e74c3c" if pct > 60 else "#f39c12" if pct > 35 else "#10b981",
                         "thickness": 0.25},
                "steps": [
                    {"range": [0,  35], "color": "#d1fae5"},
                    {"range": [35, 60], "color": "#fef3c7"},
                    {"range": [60, 100],"color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#1e293b", "width": 3},
                              "thickness": 0.8, "value": pct},
            },
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk verdict
        if pct > 60:
            st.markdown(f'<div class="risk-high"><b>🔴 High churn risk ({pct:.0f}%)</b><br>This customer is very likely to leave. Immediate retention action recommended — consider a discount, contract upgrade offer, or proactive support outreach.</div>', unsafe_allow_html=True)
        elif pct > 35:
            st.markdown(f'<div class="risk-medium"><b>🟡 Medium churn risk ({pct:.0f}%)</b><br>Monitor this customer. A loyalty reward or check-in call could tip them to stay. Watch for service complaints.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><b>🟢 Low churn risk ({pct:.0f}%)</b><br>This customer is likely to stay. Focus retention resources elsewhere.</div>', unsafe_allow_html=True)

        st.divider()

        # Key risk factors for this customer
        st.markdown("#### Why this score? Key risk factors")

        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append(("Contract type",    "Month-to-month = 3× higher churn",  "high"))
        if internet == "Fiber optic":
            risk_factors.append(("Internet service", "Fiber optic users churn more",       "medium"))
        if tech_support == "No":
            risk_factors.append(("Tech support",     "No tech support = higher churn risk","medium"))
        if online_sec == "No":
            risk_factors.append(("Online security",  "No security addon = higher risk",    "medium"))
        if tenure < 12:
            risk_factors.append(("Tenure",           f"Only {tenure}m — new customers churn most", "high"))
        if payment == "Electronic check":
            risk_factors.append(("Payment method",   "Electronic check = highest churn method", "medium"))
        if not risk_factors:
            risk_factors.append(("Profile",          "Low-risk profile across all features", "low"))

        for factor, reason, level in risk_factors:
            color = "#e74c3c" if level == "high" else "#f39c12" if level == "medium" else "#10b981"
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:6px 12px;margin:6px 0;'
                f'border-radius:0 6px 6px 0;background:var(--background-color)">'
                f'<b>{factor}</b> — {reason}</div>',
                unsafe_allow_html=True
            )

        st.divider()

        # Business value
        monthly_val = monthly
        lifetime    = monthly_val * (72 - tenure)
        st.markdown("#### Business value at stake")
        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly revenue", f"${monthly_val:.0f}")
        c2.metric("Potential lifetime value", f"${lifetime:,.0f}")
        c3.metric("Churn probability", f"{pct:.0f}%")

    else:
        st.markdown("### How this works")
        st.info("Fill in the customer profile on the left and click **Predict churn risk** to get an instant churn probability score with explanation.")

        st.markdown("#### Model performance on 1,407 held-out customers")
        metrics_df = pd.DataFrame({
            "Model":     ["Logistic Regression", "Random Forest", "XGBoost"],
            "AUC":       [0.8346, 0.8143, 0.8345],
            "Precision": [0.627,  0.611,  0.618],
            "Recall":    [0.567,  0.521,  0.539],
        })
        st.dataframe(metrics_df.style.highlight_max(subset=["AUC"], color="#d1fae5"),
                     hide_index=True, use_container_width=True)

        st.markdown("#### Dataset overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total customers", "7,032")
        col2.metric("Churn rate",      "26.6%")
        col3.metric("Features used",   "19")
        col4.metric("Best AUC",        "0.8346")

        st.markdown("#### Top churn drivers (from XGBoost)")
        drivers = pd.DataFrame({
            "Driver":    ["Contract type", "Tenure", "Monthly charges",
                          "Internet service", "Tech support", "Online security"],
            "Impact":    ["Very high", "Very high", "High", "High", "Medium", "Medium"],
            "Direction": ["Month-to-month = 3× risk", "Short tenure = high risk",
                          "High charges = higher risk", "Fiber = higher churn",
                          "No support = higher risk", "No security = higher risk"],
        })
        st.dataframe(drivers, hide_index=True, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built by **Veda Sai Polisetty** · "
    "Dataset: IBM Telco Customer Churn (Kaggle) · "
    "Stack: Python · scikit-learn · XGBoost · Streamlit · Plotly"
)
