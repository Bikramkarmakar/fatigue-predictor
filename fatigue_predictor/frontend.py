"""
frontend.py  –  Streamlit UI
-----------------------------
A self-contained interactive web app that:
  1. Takes three user inputs via sliders / number inputs
  2. Calls the FastAPI backend /predict endpoint
  3. Displays the predicted fatigue score with a visual gauge
  4. Shows model performance metrics in an expandable panel
  5. Provides a feature-correlation chart using the raw dataset

Run standalone (no backend) OR alongside the FastAPI server.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "bedtime_screentime_sleep_debt.csv")

FEATURES = [
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "caffeine_post_5pm_mg",
]
TARGET = "next_day_fatigue_score"

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="😴 Fatigue Predictor",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #57606a;
        margin-bottom: 1.5rem;
    }
    .score-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        border: 1.5px solid #c7d7fc;
    }
    .score-number {
        font-size: 4rem;
        font-weight: 900;
        color: #3b5bdb;
    }
    .score-label {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-top: 0.3rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .metric-box {
        background: #f7f8fa;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        flex: 1;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3b5bdb;
    }
    .metric-name {
        font-size: 0.78rem;
        color: #57606a;
    }
    footer {visibility: hidden;}
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

# ── Sidebar  – Inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📱 Enter Tonight's Habits")
    st.caption("Adjust the sliders or type values directly.")

    bedtime_phone_minutes = st.slider(
        "📱 Bedtime Phone Usage (minutes)",
        min_value=0,
        max_value=300,
        value=60,
        step=1,
        help="How many minutes did you use your phone in bed before sleeping?",
    )
    screen_brightness_pct = st.slider(
        "💡 Screen Brightness (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        help="Average brightness of your screen during bedtime phone usage.",
    )
    caffeine_post_5pm_mg = st.slider(
        "☕ Caffeine after 5 PM (mg)",
        min_value=0,
        max_value=400,
        value=50,
        step=5,
        help="Total caffeine consumed after 5 PM (coffee ~95 mg, tea ~47 mg, energy drink ~80–160 mg).",
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Fatigue", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("**Reference caffeine values:**")
    st.caption("☕ Espresso: ~63 mg")
    st.caption("☕ Drip coffee: ~95 mg")
    st.caption("🍵 Black tea: ~47 mg")
    st.caption("🥤 Energy drink: ~80–160 mg")


# ── Helper: gauge chart ────────────────────────────────────────────────────────
def make_gauge(score: float) -> go.Figure:
    if score <= 2.5:
        color = "#22c55e"
    elif score <= 4.5:
        color = "#84cc16"
    elif score <= 6.5:
        color = "#f59e0b"
    elif score <= 8.5:
        color = "#f97316"
    else:
        color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 52, "color": color}, "suffix": " / 10"},
            gauge={
                "axis": {"range": [1, 10], "tickwidth": 1, "tickcolor": "#aaa"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [1, 2.5], "color": "#d1fae5"},
                    {"range": [2.5, 4.5], "color": "#ecfccb"},
                    {"range": [4.5, 6.5], "color": "#fef9c3"},
                    {"range": [6.5, 8.5], "color": "#ffedd5"},
                    {"range": [8.5, 10], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
    )
    return fig


# ── Helper: scatter plots ──────────────────────────────────────────────────────
def feature_scatter(df: pd.DataFrame, feature: str, highlight_x: float):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[feature],
            y=df[TARGET],
            mode="markers",
            marker=dict(size=5, color="#93c5fd", opacity=0.6),
            name="Dataset",
        )
    )
    fig.add_vline(
        x=highlight_x,
        line_dash="dash",
        line_color="#3b5bdb",
        annotation_text="Your value",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title=feature.replace("_", " ").title(),
        yaxis_title="Fatigue Score",
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f7f8fa",
    )
    return fig


# ── Main layout ────────────────────────────────────────────────────────────────
col_result, col_charts = st.columns([1, 1.6], gap="large")

with col_result:
    st.subheader("🎯 Prediction Result")

    if predict_btn:
        with st.spinner("Calling prediction API…"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "bedtime_phone_minutes": bedtime_phone_minutes,
                        "screen_brightness_pct": screen_brightness_pct,
                        "caffeine_post_5pm_mg": caffeine_post_5pm_mg,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                score = data["next_day_fatigue_score"]
                label = data["fatigue_label"]
                note = data["confidence_note"]

                st.plotly_chart(make_gauge(score), use_container_width=True)
                st.markdown(
                    f'<div class="score-card">'
                    f'<div class="score-label">{label}</div>'
                    f'<div style="font-size:0.85rem;color:#57606a;margin-top:0.5rem">{note}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Personalized advice
                st.markdown("#### 💡 Personalized Tips")
                tips = []
                if bedtime_phone_minutes > 60:
                    tips.append("📵 Try reducing bedtime phone usage to under 30 minutes.")
                if screen_brightness_pct > 50:
                    tips.append("🌙 Lower your screen brightness or enable Night Mode.")
                if caffeine_post_5pm_mg > 100:
                    tips.append("🚫 Avoid caffeine after 5 PM — it lingers for 5–7 hours.")
                if not tips:
                    tips.append("✅ Great habits! Keep it up for optimal recovery.")
                for tip in tips:
                    st.success(tip)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot reach the API. Make sure the backend is running:\n\n"
                    "```bash\nuvicorn app:app --reload\n```"
                )
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
    else:
        st.info("👈 Set your habits in the sidebar and click **Predict Fatigue**.")
        st.plotly_chart(make_gauge(5.0), use_container_width=True)
        st.caption("Example gauge — run a prediction to see your real result.")

with col_charts:
    st.subheader("📊 Dataset Exploration")

    try:
        df = pd.read_csv(DATA_PATH)

        tab1, tab2, tab3 = st.tabs(
            ["📱 Phone Minutes", "💡 Brightness", "☕ Caffeine"]
        )
        input_vals = {
            "bedtime_phone_minutes": bedtime_phone_minutes,
            "screen_brightness_pct": screen_brightness_pct,
            "caffeine_post_5pm_mg": caffeine_post_5pm_mg,
        }
        tabs = [tab1, tab2, tab3]
        for tab, feat in zip(tabs, FEATURES):
            with tab:
                st.plotly_chart(
                    feature_scatter(df, feat, input_vals[feat]),
                    use_container_width=True,
                )
                corr = df[[feat, TARGET]].corr().iloc[0, 1]
                st.caption(f"Pearson correlation with fatigue: **{corr:.3f}**")

    except FileNotFoundError:
        st.warning("Dataset file not found — scatter plots unavailable.")

# ── Model metrics expander ─────────────────────────────────────────────────────
with st.expander("🔬 Model Performance Metrics", expanded=False):
    try:
        resp = requests.get(f"{API_URL}/model-metrics", timeout=5)
        resp.raise_for_status()
        m = resp.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{m.get('mae', '—')}")
        c2.metric("RMSE", f"{m.get('rmse', '—')}")
        c3.metric("R²", f"{m.get('r2', '—')}")
        c4.metric("CV R² (5-fold)", f"{m.get('cv_r2_mean', '—')} ± {m.get('cv_r2_std', '—')}")

        st.markdown("**Feature Importances**")
        fi = m.get("feature_importances", {})
        fi_df = pd.DataFrame(
            {"Feature": list(fi.keys()), "Importance": list(fi.values())}
        ).sort_values("Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature"))

        st.json(m)

    except Exception:
        st.info("Start the backend to load live model metrics.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Streamlit](https://streamlit.io) · "
    "ML powered by scikit-learn GradientBoostingRegressor · "
    "Dataset: bedtime_screentime_sleep_debt.csv"
)
