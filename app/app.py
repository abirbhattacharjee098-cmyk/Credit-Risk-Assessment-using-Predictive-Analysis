"""
app/app.py  –  Credit Risk Assessment  |  Streamlit UI
=======================================================
Run:
    streamlit run app/app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

# Allow imports from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.12);
}

/* ── Cards / metric boxes ── */
.metric-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: transform .2s;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-label { font-size: 13px; color: #aab4c8; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(90deg, rgba(99,102,241,0.35), rgba(236,72,153,0.25));
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.hero p  { color: #c0c8e0; margin: 8px 0 0; font-size: 1.05rem; }

/* ── Risk badge ── */
.badge-low  { background: #10b981; color:#fff; border-radius:999px; padding:6px 22px; font-weight:700; font-size:1.1rem; }
.badge-high { background: #ef4444; color:#fff; border-radius:999px; padding:6px 22px; font-weight:700; font-size:1.1rem; }

/* ── Streamlit overrides ── */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: .5px;
    transition: opacity .2s, transform .15s;
}
.stButton>button:hover { opacity:.88; transform:scale(1.01); }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #c0c8e0 !important; font-size:14px; }

h2, h3 { color: #e0e6f8; }
hr { border-color: rgba(255,255,255,0.12); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")

MODEL_FILES = {
    "🧠 Random Forest":       "random_forest.pkl",
    "⚡ XGBoost":             "xgboost.pkl",
    "📊 Logistic Regression": "logistic_regression.pkl",
    "🌳 Decision Tree":       "decision_tree.pkl",
}

NUMERIC_FEATURES = [
    "age", "income", "loan_amount", "loan_term", "credit_score",
    "employment_years", "num_credit_lines", "debt_to_income",
    "num_delinquencies", "previous_defaults",
    "loan_to_income", "monthly_payment", "payment_to_income",
]
CATEGORICAL_FEATURES = [
    "education", "home_ownership", "loan_purpose", "employment_status",
]


@st.cache_resource
def load_model(name: str):
    path = os.path.join(MODELS_DIR, MODEL_FILES[name])
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_all_artifacts():
    """Load scaler, imputer, feature columns."""
    artefacts = {}
    for key, fname in [("scaler", "scaler.pkl"),
                       ("imputer", "imputer.pkl"),
                       ("features", "feature_columns.pkl")]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            artefacts[key] = joblib.load(p)
    return artefacts


def models_exist() -> bool:
    return any(
        os.path.exists(os.path.join(MODELS_DIR, f))
        for f in MODEL_FILES.values()
    )


def preprocess_input(raw: dict, artefacts: dict) -> np.ndarray:
    """Transform raw UI dict → scaled numpy array for model.predict_proba."""
    df = pd.DataFrame([raw])

    # Impute
    df[NUMERIC_FEATURES] = artefacts["imputer"].transform(
        df[NUMERIC_FEATURES].values
    )

    # One-hot encode
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Align
    for col in artefacts["features"]:
        if col not in df.columns:
            df[col] = 0
    df = df[artefacts["features"]]

    # Scale
    numeric_in_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    df[numeric_in_features] = artefacts["scaler"].transform(
        df[numeric_in_features].values
    )

    return df.values


def gauge_chart(probability: float) -> go.Figure:
    """Plotly gauge showing default probability."""
    color = "#ef4444" if probability >= 0.5 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        delta={"reference": 50, "valueformat": ".1f"},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar":  {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.2)"},
                {"range": [40, 60], "color": "rgba(234,179,8,0.2)"},
                {"range": [60,100], "color": "rgba(239,68,68,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": "Default Probability", "font": {"size": 18, "color": "#c0c8e0"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font={"color": "#c0c8e0"},
        height=280,
        margin=dict(t=40, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.markdown("## 💰 Credit Risk AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home & Prediction",
         "📊 EDA Dashboard",
         "📈 Model Performance",
         "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Engineered by Quantitative Risk Intelligence Group")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Home & Prediction
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & Prediction":

    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>💰 Credit Risk Assessment System</h1>
      <p>AI-powered loan default prediction using Predictive Analytics & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Check models ───────────────────────────────────────────────────────
    if not models_exist():
        st.error("""
        ⚠️  **Models not found.**

        Please run the training pipeline first:
        ```bash
        python src/pipeline.py
        ```
        """)
        st.stop()

    artefacts = load_all_artifacts()
    if not artefacts:
        st.error("Preprocessing artefacts missing. Run `python src/pipeline.py`.")
        st.stop()

    # ── Model selector ────────────────────────────────────────────────────
    st.markdown("### 🤖 Select Prediction Model")
    available = [k for k, v in MODEL_FILES.items()
                 if os.path.exists(os.path.join(MODELS_DIR, v))]
    selected_model_name = st.selectbox("", available, label_visibility="collapsed")
    model = load_model(selected_model_name)

    st.divider()

    # ── Input form ────────────────────────────────────────────────────────
    st.markdown("### 📋 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Details**")
        age   = st.slider("Age", 18, 75, 35, key="age_slider")
        education = st.selectbox("Education",
            ["High School", "Bachelor", "Master", "PhD"], key="edu_sel")
        home_ownership = st.selectbox("Home Ownership",
            ["Rent", "Own", "Mortgage"], key="home_sel")
        employment_status = st.selectbox("Employment Status",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"], key="emp_sel")
        employment_years = st.slider("Employment Years", 0, 35, 5, key="emp_yrs")

    with col2:
        st.markdown("**💵 Financial Profile**")
        income = st.number_input("Annual Income ($)", 10_000, 300_000, 60_000,
                                  step=1_000, key="income_input")
        credit_score = st.slider("Credit Score", 300, 850, 650, key="cs_slider")
        debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 0.75, 0.25,
                                    step=0.01, format="%.2f", key="dti_slider")
        num_credit_lines = st.slider("Number of Credit Lines", 1, 20, 5, key="ncl_slider")
        num_delinquencies = st.slider("Number of Delinquencies", 0, 10, 0, key="nd_slider")
        previous_defaults = st.slider("Previous Defaults", 0, 5, 0, key="pd_slider")

    with col3:
        st.markdown("**🏦 Loan Details**")
        loan_amount = st.number_input("Loan Amount ($)", 500, 100_000, 15_000,
                                       step=500, key="la_input")
        loan_term = st.selectbox("Loan Term (months)",
            [12, 24, 36, 48, 60], index=2, key="lt_sel")
        loan_purpose = st.selectbox("Loan Purpose",
            ["Debt Consolidation", "Home Improvement", "Business",
             "Medical", "Education", "Personal"], key="lp_sel")

        # ── Derived features (auto-computed) ──────────────────────────────
        loan_to_income    = round(loan_amount / (income + 1), 4)
        monthly_rate      = 0.07 / 12
        monthly_payment   = round(
            (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -loan_term), 2
        )
        payment_to_income = round(monthly_payment / (income / 12 + 1), 4)

        st.markdown("**📐 Auto-Computed Ratios**")
        st.info(f"Loan-to-Income Ratio : **{loan_to_income:.3f}**")
        st.info(f"Monthly Payment      : **${monthly_payment:,.0f}**")
        st.info(f"Payment-to-Income    : **{payment_to_income:.3f}**")

    st.divider()

    # ── Risk threshold slider ─────────────────────────────────────────────
    st.markdown("### 🎯 Risk Threshold Tuning")
    threshold = st.slider(
        "Decision Threshold (probability above which = High Risk)",
        0.20, 0.80, 0.50, step=0.05,
        help="Lower = stricter (flag more applicants as risky). Default is 0.50.",
        key="threshold_slider"
    )

    # ── Prediction button ─────────────────────────────────────────────────
    predict_btn = st.button("🔍  Predict Credit Risk", key="predict_btn")

    if predict_btn:
        raw_input = {
            "age":               age,
            "income":            float(income),
            "loan_amount":       float(loan_amount),
            "loan_term":         loan_term,
            "credit_score":      credit_score,
            "employment_years":  float(employment_years),
            "num_credit_lines":  num_credit_lines,
            "debt_to_income":    debt_to_income,
            "num_delinquencies": num_delinquencies,
            "previous_defaults": previous_defaults,
            "loan_to_income":    loan_to_income,
            "monthly_payment":   monthly_payment,
            "payment_to_income": payment_to_income,
            "education":         education,
            "home_ownership":    home_ownership,
            "loan_purpose":      loan_purpose,
            "employment_status": employment_status,
        }

        X_input = preprocess_input(raw_input, artefacts)
        probability   = model.predict_proba(X_input)[0][1]
        prediction    = 1 if probability >= threshold else 0

        # ── Result display ─────────────────────────────────────────────
        st.divider()
        st.markdown("## 🎯 Prediction Result")

        col_gauge, col_result, col_tips = st.columns([1.2, 1, 1])

        with col_gauge:
            st.plotly_chart(gauge_chart(probability), use_container_width=True)

        with col_result:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                <div style='text-align:center'>
                  <div class='badge-high'>⚠️  HIGH RISK</div>
                  <p style='margin-top:12px;color:#fca5a5;font-size:15px'>
                  This applicant is likely to <strong>default</strong> on the loan.
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align:center'>
                  <div class='badge-low'>✅  LOW RISK</div>
                  <p style='margin-top:12px;color:#6ee7b7;font-size:15px'>
                  This applicant is likely to <strong>repay</strong> the loan.
                  </p>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=[probability * 100, (1 - probability) * 100],
                y=["Default", "No Default"],
                orientation="h",
                marker_color=["#ef4444", "#10b981"],
                text=[f"{probability*100:.1f}%", f"{(1-probability)*100:.1f}%"],
                textposition="auto",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font={"color": "#c0c8e0"},
                height=140,
                margin=dict(t=0, b=0, l=0, r=0),
                xaxis=dict(range=[0,100], showgrid=False, visible=False),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_tips:
            st.markdown("#### 💡 Business Insights")
            risk_factors = []
            if credit_score < 580:
                risk_factors.append("🔴 Very low credit score < 580")
            elif credit_score < 670:
                risk_factors.append("🟡 Fair credit score (580–670)")
            if previous_defaults > 0:
                risk_factors.append(f"🔴 {previous_defaults} previous default(s)")
            if debt_to_income > 0.43:
                risk_factors.append(f"🟡 High DTI ratio ({debt_to_income:.2f})")
            if num_delinquencies > 2:
                risk_factors.append(f"🔴 {num_delinquencies} delinquencies")
            if employment_status == "Unemployed":
                risk_factors.append("🔴 Currently unemployed")
            if loan_to_income > 0.35:
                risk_factors.append("🟡 High loan-to-income ratio")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"- {rf}")
            else:
                st.markdown("✅ No major risk flags detected.")

            st.markdown(f"""
            ---
            **Model used:** {selected_model_name}
            **Threshold:** {threshold:.0%}
            **Probability:** {probability:.4f}
            """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>📊 Exploratory Data Analysis</h1>
      <p>Understanding the credit dataset through rich visualizations</p>
    </div>
    """, unsafe_allow_html=True)

    data_path = os.path.join(ROOT, "data", "credit_data.csv")
    if not os.path.exists(data_path):
        st.warning("Dataset not found. Run `python src/pipeline.py` first.")
        st.stop()

    df = pd.read_csv(data_path)

    # ── KPI cards ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpi_data = [
        (c1, "Total Applicants",   f"{len(df):,}",                               "#6366f1"),
        (c2, "Default Rate",       f"{df['default'].mean():.1%}",                 "#ef4444"),
        (c3, "Avg Credit Score",   f"{df['credit_score'].mean():.0f}",            "#10b981"),
        (c4, "Avg Loan Amount",    f"${df['loan_amount'].mean():,.0f}",           "#f59e0b"),
    ]
    for col, label, value, color in kpi_data:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value' style='color:{color}'>{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class distribution ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        counts = df["default"].value_counts()
        fig_pie = px.pie(
            values=counts.values,
            names=["No Default", "Default"],
            color_discrete_sequence=["#6366f1","#ef4444"],
            title="Loan Default Distribution",
            hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        default_rate_by_purpose = (
            df.groupby("loan_purpose")["default"].mean().reset_index()
        )
        default_rate_by_purpose.columns = ["Purpose", "Default Rate"]
        fig_bar2 = px.bar(
            default_rate_by_purpose.sort_values("Default Rate", ascending=False),
            x="Purpose", y="Default Rate",
            color="Default Rate",
            color_continuous_scale="RdYlGn_r",
            title="Default Rate by Loan Purpose",
        )
        fig_bar2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0",
                               coloraxis_showscale=False)
        st.plotly_chart(fig_bar2, use_container_width=True)

    # ── Scatter: Credit score vs Income ───────────────────────────────────
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="credit_score", y="income",
        color=sample["default"].map({0:"No Default", 1:"Default"}),
        color_discrete_map={"No Default":"#6366f1","Default":"#ef4444"},
        opacity=0.6, title="Credit Score vs Income",
        labels={"credit_score":"Credit Score","income":"Annual Income ($)"},
        size_max=8,
    )
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Numeric distributions ─────────────────────────────────────────────
    st.markdown("### 📉 Feature Distribution by Default Label")
    feat = st.selectbox("Select feature to explore",
                        ["credit_score","income","loan_amount","debt_to_income",
                         "age","employment_years","num_delinquencies","previous_defaults"],
                        key="eda_feat_sel")
    fig_hist = px.histogram(
        df, x=feat,
        color=df["default"].map({0:"No Default", 1:"Default"}),
        barmode="overlay", nbins=50,
        color_discrete_map={"No Default":"#6366f1","Default":"#ef4444"},
        opacity=0.7,
        title=f"Distribution of {feat.replace('_',' ').title()}",
    )
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.markdown("### 🔥 Correlation Heatmap")
    num_cols = ["age","income","loan_amount","credit_score","debt_to_income",
                "employment_years","num_delinquencies","previous_defaults",
                "loan_to_income","default"]
    corr = df[num_cols].corr()
    fig_hm = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
    )
    fig_hm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0",
                         height=500)
    st.plotly_chart(fig_hm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Model Performance
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="hero">
      <h1>📈 Model Performance Dashboard</h1>
      <p>Compare all trained models across key evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)

    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    if not os.path.exists(summary_path):
        st.warning("Run `python src/pipeline.py` first to generate metrics.")
        st.stop()

    results = pd.read_csv(summary_path)

    # ── KPIs ──────────────────────────────────────────────────────────────
    best_row = results.loc[results["ROC-AUC"].idxmax()]
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Best Model",    best_row["Model"],          "#6366f1"),
        (c2, "Best ROC-AUC",  f"{best_row['ROC-AUC']:.4f}","#10b981"),
        (c3, "Best F1-Score", f"{results['F1-Score'].max():.4f}","#f59e0b"),
        (c4, "Best Accuracy", f"{results['Accuracy'].max():.4f}","#ec4899"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value' style='color:{color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics table ──────────────────────────────────────────────────────
    st.markdown("### 📋 Metrics Summary Table")
    styled = results.style.background_gradient(
        subset=["ROC-AUC","F1-Score","Recall"], cmap="RdYlGn"
    ).format({"Accuracy":"{:.4f}","Precision":"{:.4f}",
              "Recall":"{:.4f}","F1-Score":"{:.4f}","ROC-AUC":"{:.4f}"})
    st.dataframe(results, use_container_width=True, hide_index=True)

    # ── Grouped bar chart ──────────────────────────────────────────────────
    metric_cols = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    melted = results.melt(id_vars="Model", value_vars=metric_cols,
                          var_name="Metric", value_name="Score")
    fig_compare = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#6366f1","#ec4899","#10b981","#f59e0b"],
        title="Model Comparison – All Metrics",
    )
    fig_compare.update_layout(
        yaxis=dict(range=[0.5, 1.0]),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0",
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────
    st.markdown("### 🕸️ Radar Chart – Model Profiles")
    colors = ["#6366f1","#ec4899","#10b981","#f59e0b"]
    fig_radar = go.Figure()
    cats = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    for (_, row), color in zip(results.iterrows(), colors):
        vals = [row[c] for c in cats] + [row[cats[0]]]  # close loop
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=cats + [cats[0]],
            fill="toself",
            name=row["Model"],
            line=dict(color=color),
            fillcolor=color.replace("#","rgba(") + ",0.12)",
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5,1.0],
                                   color="#aab4c8")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#c0c8e0",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Saved report images ────────────────────────────────────────────────
    st.markdown("### 🖼️ Generated Report Plots")
    image_map = {
        "ROC Curves":             "roc_curves.png",
        "Confusion Matrices":     "confusion_matrices.png",
        "Feature Importance (RF)":"feature_importance_random_forest.png",
        "Feature Importance (XGB)":"feature_importance_xgboost.png",
    }
    tabs = st.tabs(list(image_map.keys()))
    for tab, (label, fname) in zip(tabs, image_map.items()):
        with tab:
            path = os.path.join(REPORTS_DIR, fname)
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Plot not found: {fname}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – About
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown("""
    <div class="hero">
      <h1>ℹ️ About This Project</h1>
      <p>Advanced Credit Risk Assessment using Machine Learning – Enterprise Analytics Solution</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview
    This system predicts whether a loan applicant is likely to **default** on their
    loan using historical financial and demographic data.

    ## 📦 Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.10+ |
    | ML Framework | Scikit-learn, XGBoost |
    | Imbalance Handling | SMOTE (imbalanced-learn) |
    | UI | Streamlit |
    | Visualization | Plotly, Seaborn, Matplotlib |
    | Data | Pandas, NumPy |

    ## 🤖 Models Implemented
    | Model | Key Strength |
    |-------|-------------|
    | Logistic Regression | Baseline, interpretable |
    | Decision Tree | Simple, explainable rules |
    | Random Forest | High accuracy, robust |
    | XGBoost | Best performance, gradient boosting |

    ## 📐 Feature Engineering
    - **Loan-to-Income Ratio** = Loan Amount / Annual Income
    - **Monthly Payment** = Computed via loan amortization formula
    - **Payment-to-Income** = Monthly Payment / Monthly Income
    - **SMOTE** applied to training data to handle ~30% default imbalance

    ## 📊 Evaluation Metrics
    - **Accuracy** – Overall correct predictions
    - **Precision** – Of predicted positives, how many are truly positive
    - **Recall** – Of all true positives, how many were caught
    - **F1-Score** – Harmonic mean of Precision and Recall
    - **ROC-AUC** – Area under the Receiver Operating Characteristic curve

    ## 🚀 How to Run
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train models (one-time)
    python src/pipeline.py

    # 3. Launch the app
    streamlit run app/app.py
    ```

    ## 📁 Project Structure
    ```
    Credit-Risk-Analytics-Platform/
    ├── data/               ← Dataset (auto-generated)
    ├── src/
    │   ├── data_loader.py
    │   ├── preprocessing.py
    │   ├── eda.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── pipeline.py     ← Run this first!
    ├── models/             ← Trained .pkl files
    ├── reports/            ← EDA & evaluation plots
    ├── app/
    │   └── app.py          ← This Streamlit app
    ├── requirements.txt
    └── README.md
    ```
    """)
