# Fatigue Predictor – Quick-Start README

## 😴 Next-Day Fatigue Predictor

A machine learning web application that predicts your **next-day fatigue score (1–10)**
from three bedtime inputs:
- 📱 **Bedtime phone usage** (minutes)
- 💡 **Screen brightness** (%)
- ☕ **Caffeine after 5 PM** (mg)

---

## Project Structure

```
fatigue_predictor/
├── app_standalone.py   ← ✅ RECOMMENDED: All-in-one Streamlit app (deploy this)
├── frontend.py         ← Streamlit UI (requires backend running)
├── app.py              ← FastAPI backend (REST API)
├── train_model.py      ← Model training script
├── startup.py          ← One-command launcher (backend + frontend)
├── requirements.txt    ← All Python dependencies
└── .streamlit/
    └── config.toml     ← Streamlit configuration
bedtime_screentime_sleep_debt.csv  ← Dataset (project root)
```

---

## Option A – Streamlit Community Cloud (Shareable Public URL) ⭐

This is the **easiest way to share** with others. Streamlit Community Cloud is free.

### Steps
1. **Create a free GitHub account** at https://github.com if you don't have one.
2. **Create a new public repository** (e.g. `fatigue-predictor`).
3. **Upload these files** to the root of the repo:
   - `fatigue_predictor/app_standalone.py`
   - `fatigue_predictor/requirements.txt`
   - `fatigue_predictor/.streamlit/config.toml`
   - `bedtime_screentime_sleep_debt.csv`
4. Go to **https://share.streamlit.io** → Sign in with GitHub.
5. Click **"New app"** → Select your repo, branch `main`, main file path:
   ```
   fatigue_predictor/app_standalone.py
   ```
6. Click **"Deploy"** — takes ~2 minutes.
7. 🎉 You get a public URL like `https://yourname-fatigue.streamlit.app` — share it!

> **Note:** The dataset CSV must be at the project root of the repo, one level above `fatigue_predictor/`.

---

## Option B – Run Locally

### 1. Install dependencies
```bash
cd fatigue_predictor
pip install -r requirements.txt
```

### 2. Train the model (first time only)
```bash
python train_model.py
```
This creates `model.pkl`, `scaler.pkl`, and `model_metrics.json`.

### 3. Start everything with one command
```bash
python startup.py
```
Opens:
- FastAPI backend at http://localhost:8000
- Streamlit frontend at http://localhost:8501

### Or start separately
**Backend:**
```bash
uvicorn app:app --reload --port 8000
```
**Frontend:**
```bash
streamlit run frontend.py
```

**All-in-one (no backend needed):**
```bash
streamlit run app_standalone.py
```

---

## API Endpoints (Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict fatigue score |
| `GET` | `/health` | Liveness check |
| `GET` | `/model-metrics` | Training metrics |

### Example `/predict` request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bedtime_phone_minutes": 90, "screen_brightness_pct": 70, "caffeine_post_5pm_mg": 120}'
```

### Example response
```json
{
  "next_day_fatigue_score": 7.43,
  "fatigue_label": "High Fatigue 😴",
  "confidence_note": "Model R² = 0.87"
}
```

---

## Model

| Property | Value |
|----------|-------|
| Algorithm | Gradient Boosting Regressor |
| Features | 3 (phone mins, brightness, caffeine) |
| Target | next_day_fatigue_score (1–10) |
| Train/Test split | 80% / 20% |
| Cross-validation | 5-fold |

---

## License
For educational and demonstration purposes.
