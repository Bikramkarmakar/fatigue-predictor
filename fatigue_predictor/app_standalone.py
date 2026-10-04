"""
app_standalone.py  –  All-in-one Streamlit app (no separate backend needed)
-----------------------------------------------------------------------------
This version embeds the ML inference directly inside Streamlit so that it can
be deployed to Streamlit Community Cloud with a single file.

Deployment steps:
  1. Push this project to a public GitHub repo.
  2. Go to https://share.streamlit.io  → "New app"
  3. Select repo, branch "main", and set main file path:
         fatigue_predictor/app_standalone.py
  4. Click "Deploy" — Streamlit installs requirements.txt automatically.
  5. Share the generated public URL (e.g. https://yourname-fatigue.streamlit.app)
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "bedtime_screentime_sleep_debt.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

FEATURES = ["bedtime_phone_minutes", "screen_brightness_pct", "caffeine_post_5pm_mg"]
TARGET = "next_day_fatigue_score"

# ── Train or load model (cached so Streamlit only trains once) ────────────────
@st.cache_resource(show_spinner="🔧 Training model on first run…")
def get_model():
    """Train model if artefacts missing; otherwise load from disk."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        return model, scaler, metrics

    # ---- Train fresh ----
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_sc, y, cv=5, scoring="r2")

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "n_samples": int(len(df)),
        "model": "GradientBoostingRegressor",
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURES, model.feature_importances_)
        },
    }

    # Persist so subsequent runs skip training
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return model, scaler, metrics


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# ── Load artefacts ─────────────────────────────────────────────────────────────
MODEL, SCALER, METRICS = get_model()

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="😴 Fatigue Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {font-size:2.4rem;font-weight:800;color:#1f2328;margin-bottom:0.2rem;}
    .subtitle   {font-size:1.05rem;color:#57606a;margin-bottom:1.5rem;}
    .score-card {background:#f0f4ff;border-radius:12px;padding:1.5rem 2rem;
                 text-align:center;border:1.5px solid #c7d7fc;}
    .score-label{font-size:1.3rem;font-weight:600;color:#333;margin-top:0.3rem;}
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">😴 Next-Day Fatigue Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict how tired you will feel tomorrow based on your bedtime habits tonight.</p>',
    unsafe_allow_html=True,
)
st.divider()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fatigue_label(score: float) -> str:
    if score <= 2.5:   return "Very Low Fatigue 😊"
    elif score <= 4.5: return "Low Fatigue 🙂"
    elif score <= 6.5: return "Moderate Fatigue 😐"
    elif score <= 8.5: return "High Fatigue 😴"
    else:              return "Severe Fatigue 🥴"


def make_gauge(score: float) -> go.Figure:
    colors = {(1, 2.5): "#22c55e", (2.5, 4.5): "#84cc16",
              (4.5, 6.5): "#f59e0b", (6.5, 8.5): "#f97316", (8.5, 10): "#ef4444"}
    bar_color = "#ef4444"
    for (lo, hi), c in colors.items():
        if lo <= score <= hi:
            bar_color = c
            break
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": bar_color}, "suffix": " / 10"},
        gauge={
            "axis": {"range": [1, 10], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [1, 2.5],  "color": "#d1fae5"},
                {"range": [2.5, 4.5],"color": "#ecfccb"},
                {"range": [4.5, 6.5],"color": "#fef9c3"},
                {"range": [6.5, 8.5],"color": "#ffedd5"},
                {"range": [8.5, 10], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 4},
                          "thickness": 0.75, "value": score},
        },
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", height=260)
    return fig


def feature_scatter(df, feature, highlight_x):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[feature], y=df[TARGET], mode="markers",
        marker=dict(size=5, color="#93c5fd", opacity=0.6), name="Dataset",
    ))
    fig.add_vline(x=highlight_x, line_dash="dash", line_color="#3b5bdb",
                  annotation_text="Your value", annotation_position="top right")
    fig.update_layout(
        xaxis_title=feature.replace("_", " ").title(),
        yaxis_title="Fatigue Score",
        margin=dict(l=10, r=10, t=30, b=10),
        height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f7f8fa",
    )
    return fig


def run_prediction(phone_min, brightness, caffeine):
    features = np.array([[phone_min, brightness, caffeine]])
    scaled = SCALER.transform(features)
    raw = float(MODEL.predict(scaled)[0])
    return round(max(1.0, min(10.0, raw)), 2)


# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📱 Enter Tonight's Habits")
    st.caption("Adjust sliders, then click **Predict**.")

    bedtime_phone_minutes = st.slider(
        "📱 Bedtime Phone Usage (minutes)", 0, 300, 60, 1,
        help="Minutes on phone in bed before sleeping.",
    )
    screen_brightness_pct = st.slider(
        "💡 Screen Brightness (%)", 0, 100, 60, 1,
        help="Average screen brightness during bedtime usage.",
    )
    caffeine_post_5pm_mg = st.slider(
        "☕ Caffeine after 5 PM (mg)", 0, 400, 50, 5,
        help="Total caffeine after 5 PM (coffee ≈95 mg, tea ≈47 mg).",
    )
    st.markdown("---")
    predict_btn = st.button("🔮 Predict Fatigue", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("**Caffeine reference:**")
    st.caption("☕ Espresso: ~63 mg | Drip coffee: ~95 mg")
    st.caption("🍵 Black tea: ~47 mg | 🥤 Energy drink: ~80–160 mg")


# ── Main columns ───────────────────────────────────────────────────────────────
col_result, col_charts = st.columns([1, 1.6], gap="large")

with col_result:
    st.subheader("🎯 Prediction Result")

    if predict_btn:
        score = run_prediction(bedtime_phone_minutes, screen_brightness_pct, caffeine_post_5pm_mg)
        label = _fatigue_label(score)

        st.plotly_chart(make_gauge(score), use_container_width=True)
        st.markdown(
            f'<div class="score-card">'
            f'<div class="score-label">{label}</div>'
            f'<div style="font-size:0.85rem;color:#57606a;margin-top:0.5rem">'
            f'Model R² = {METRICS.get("r2", "N/A")}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 💡 Personalized Tips")
        tips = []
        if bedtime_phone_minutes > 60:
            tips.append("📵 Reduce bedtime phone usage to under 30 minutes.")
        if screen_brightness_pct > 50:
            tips.append("🌙 Lower brightness or enable Night Mode / blue-light filter.")
        if caffeine_post_5pm_mg > 100:
            tips.append("🚫 Cut caffeine after 5 PM — it stays active for 5–7 hours.")
        if not tips:
            tips.append("✅ Great habits! You're set for optimal recovery.")
        for tip in tips:
            st.success(tip)
    else:
        st.info("👈 Set your habits in the sidebar and click **Predict Fatigue**.")
        st.plotly_chart(make_gauge(5.0), use_container_width=True)
        st.caption("Example gauge — run a prediction to see your real result.")

with col_charts:
    st.subheader("📊 Dataset Exploration")
    try:
        df = load_dataset()
        tab1, tab2, tab3 = st.tabs(["📱 Phone Minutes", "💡 Brightness", "☕ Caffeine"])
        input_vals = {
            "bedtime_phone_minutes": bedtime_phone_minutes,
            "screen_brightness_pct": screen_brightness_pct,
            "caffeine_post_5pm_mg": caffeine_post_5pm_mg,
        }
        for tab, feat in zip([tab1, tab2, tab3], FEATURES):
            with tab:
                st.plotly_chart(feature_scatter(df, feat, input_vals[feat]),
                                use_container_width=True)
                corr = df[[feat, TARGET]].corr().iloc[0, 1]
                st.caption(f"Pearson correlation with fatigue: **{corr:.3f}**")
    except FileNotFoundError:
        st.warning("Dataset not found — scatter plots unavailable.")


# ── Model metrics ──────────────────────────────────────────────────────────────
with st.expander("🔬 Model Performance Metrics", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",  METRICS.get("mae", "—"))
    c2.metric("RMSE", METRICS.get("rmse", "—"))
    c3.metric("R²",   METRICS.get("r2", "—"))
    c4.metric("CV R² (5-fold)",
              f"{METRICS.get('cv_r2_mean','—')} ± {METRICS.get('cv_r2_std','—')}")

    fi = METRICS.get("feature_importances", {})
    if fi:
        fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
        st.bar_chart(fi_df.set_index("Feature"))

    st.json(METRICS)

# ── Distribution chart ─────────────────────────────────────────────────────────
with st.expander("📈 Fatigue Score Distribution in Dataset", expanded=False):
    try:
        df = load_dataset()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[TARGET], nbinsx=20,
            marker_color="#93c5fd", opacity=0.8,
            name="Fatigue Score",
        ))
        fig.update_layout(
            xaxis_title="Fatigue Score", yaxis_title="Count",
            bargap=0.05, height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f7f8fa",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Dataset size: **{len(df):,} records** | "
                   f"Mean fatigue: **{df[TARGET].mean():.2f}** | "
                   f"Std: **{df[TARGET].std():.2f}**")
    except FileNotFoundError:
        st.warning("Dataset not found.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "ML: scikit-learn GradientBoostingRegressor · "
    "Dataset: bedtime_screentime_sleep_debt.csv"
)
