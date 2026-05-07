# Probabilistic Flight Delay Forecasting with Cross-Airport Delay Propagation Analysis

## 🌐 Live App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flightdelayforecasting.streamlit.app/)


> **Authors:** Aarsh Adhvaryu · Nikita Sharma · Ram Sharma  
> **Data:** US Bureau of Transportation Statistics (BTS) · January 2023 – December 2025  
> **Scope:** 10 busiest US hub airports · ~22 million raw flight records

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Questions](#research-questions)
3. [Repository Structure](#repository-structure)
4. [Installation & Setup](#installation--setup)
5. [Running the Project](#running-the-project)
6. [Data Pipeline](#data-pipeline)
7. [Models Implemented](#models-implemented)
8. [Key Results](#key-results)
9. [Dashboard Guide](#dashboard-guide)
10. [Reproducibility](#reproducibility)
11. [Dependencies](#dependencies)

---

## Project Overview

Flight delays at hub airports do not occur in isolation — they cascade across the network through shared aircraft rotations, crews, and passengers. Current operational systems issue single-point forecasts and treat airports independently, providing no quantification of how wrong those forecasts might be.

This project builds a **probabilistic forecasting pipeline** that:

- Provides **calibrated prediction intervals** via Quantile Recurrent Neural Networks (QRNN) trained with Pinball Loss
- Models **cross-airport delay propagation** through Granger causality testing on daily/hourly aggregated delays and physical tail-number aircraft chain analysis
- Delivers an **operational hourly alert system** that fires when a source hub's rolling 3-hour average delay exceeds a configurable threshold, predicting downstream impacts using statistically validated lag windows

The target variable is the **daily mean departure delay (minutes) per airport**, benchmarked against the US BTS standard (≥15 min = delay).

---

## Research Questions

| # | Question | Finding |
|---|----------|---------|
| **RQ1** | Are ARIMA/LSTM residuals Gaussian? | ❌ Rejected for all 10 airports, all 4 models (Shapiro-Wilk p < 10⁻¹², skewness 1.3–3.7, excess kurtosis 4–22). Log-Normal (shifted) provides best fit (AIC 3600 vs Normal 3793). |
| **RQ2** | Does QRNN achieve PICP ≥ 0.90? | ⚠️ Partially: range 0.717–0.855. Best: PHX 0.855, ORD 0.852. Temperature scaling applied at under-covering airports. |
| **RQ3** | Do hub airports exhibit significant delay propagation? | ✅ Yes — 68/90 Granger pairs significant (75.6%), mean tail-chain carry-over 74.2%, hourly alert system macro F1 = **0.760** (vs 0.364 daily — 2× improvement). |

---

## Repository Structure

```
.
├── flight_delay.ipynb          # Main analysis notebook (Cells 1–116)
├── dash.py                     # Streamlit interactive dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── dashboard                   # Results
├── ppt.pdf
├── runtime.txt
└── Flight_Delay_report.docx



```

### Notebook Cell Map

| Cell Range | Stage |
|-----------|-------|
| 2–3 | Data ingestion (parquet load, 22M rows) |
| 5–6 | Cleaning (drop cancelled, clip delays, fill NaNs) |
| 8 | Global config (TOP_N=10, TEST_DAYS=365, SEED=42) |
| 11–14 | Daily aggregation + assertion guards |
| 17–25 | Exploratory Data Analysis (EDA) — heatmaps, delay causes |
| 27–29 | STL decomposition — seasonal/trend strength |
| 31–33 | Stationarity (ADF), ACF/PACF process identification |
| 35–38 | Feature engineering (23 features) |
| 40–41 | Train/test split (516 train / 366 test days) |
| 43–49 | Baseline models + point forecast metrics |
| 52–55 | ARIMA (auto_arima, non-seasonal) |
| 57–59 | SARIMA (auto_arima, m=7 weekly) |
| 61–66 | SARIMAX + exogenous feature selection |
| 72–78 | LSTM (hidden=64, seq=21) |
| 80–82 | GRU (hidden=64, 17K params) |
| 84–89 | QRNN (Pinball Loss, 7 quantiles, temperature scaling) |
| 92–95 | Master loop — all models × 10 airports |
| 96–101 | Residual noise analysis — normality + distribution fitting |
| 102–106 | Tail-number flight chain analysis |
| 107–108 | Daily & hourly wide matrices |
| 109–112 | Granger causality (68 significant pairs) |
| 113–116 | Hourly alert system evaluation (F1 = 0.760) |

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip or conda

### 1. Clone / Download

```bash
git clone <repo-url>
cd flight-delay-forecasting
```

### 2. Create Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Data

Obtain the BTS On-Time Performance files (Parquet format) for January 2023 – December 2025 from:

```
https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ
```

Fields required: `FL_DATE`, `OP_CARRIER`, `TAIL_NUM`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `DEP_TIME`, `DEP_DELAY`, `CANCELLED`, `DIVERTED`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY`, `CANCELLATION_CODE`

Place all downloaded `.parquet` files in a `data/` directory and update the `DATA_PATH` variable in Cell 2 of the notebook.

---

## Running the Project

### Jupyter Notebook (full analysis)

```bash
jupyter notebook flight_delay.ipynb
# or
jupyter lab flight_delay.ipynb
```

Run cells sequentially from top to bottom. The notebook is designed to be fully reproducible with `SEED = 42`.

> **Estimated runtime:** ~45–90 minutes on CPU (GRU/LSTM/QRNN training). GPU significantly reduces deep learning stages.

### Streamlit Dashboard

```bash
streamlit run dash.py
```

Opens at `http://localhost:8501`. The dashboard uses pre-computed metrics from the notebook and generates synthetic demo time-series for visualization — no live data connection is required.

---

## Data Pipeline

```
Raw BTS Parquet (22,085,189 rows)
        ↓
  Drop CANCELLED / DIVERTED
  Drop NULL DEP_DELAY
  Clip DEP_DELAY to [−60, 600] min
  Fill delay-cause NaN → 0
  Derive DAY_OF_WEEK from FL_DATE
  Drop CANCELLATION_CODE
        ↓
  Filter to TOP 10 hubs by departure volume:
  ATL · DEN · DFW · ORD · CLT · LAX · PHX · LAS · SEA · MCO
        ↓
  Aggregate: daily mean DEP_DELAY per airport
  (7,255,696 hub-filtered rows → 8,819 daily rows × 10 cols)
        ↓
  Feature Engineering (23 features):
  • Cyclical: sin/cos month, day-of-week
  • Lags: lag_1, lag_2, lag_3, lag_7, lag_14
  • Rolling: mean_7, mean_14, mean_30; std_7
  • Calendar: is_weekend, is_monday, is_friday
  • Seasonal: is_summer (Jun–Aug), is_holiday_season (Nov–Dec)
  • pct_delayed (share of flights delayed >15 min)
        ↓
  Train / Test Split (no leakage):
  Train: 2023-01-01 – 2024-05-31 (516 days)
  Test:  2024-06-01 – 2025-05-31 (366 days)
```

---

## Models Implemented

### Baseline Models

| Model | Rule |
|-------|------|
| Arithmetic Mean | Always predict training mean |
| Naïve | Last observed value |
| Seasonal Naïve | Same weekday last week (lag-7) |
| Moving Average (k=7) | Mean of last 7 observations |

### Statistical Models

| Model | Specification | PHX RMSE |
|-------|--------------|----------|
| ARIMA | (0,1,2) — auto-selected via AIC | 5.247 min |
| SARIMA | (1,0,2)×(0,1,1)[7] — weekly seasonal | **4.896 min** ← Champion |
| SARIMAX | SARIMA + {weather_lag1, is_summer, is_weekend} | 4.935 min |

### Deep Learning Models

| Model | Architecture | PHX RMSE |
|-------|-------------|----------|
| LSTM | 2-layer, hidden=64, seq_len=21, early stop | 5.585 min |
| GRU | 1-layer, hidden=64, 17K params, early stop | 4.985 min |
| QRNN | Pinball Loss, 7 quantiles (0.05–0.95), T-scaling | 5.250 min (median) |

### Evaluation Metrics

**Point forecasts:** RMSE, MAE, MAPE, sMAPE  
**Probabilistic:** PICP (target ≥ 0.90), PINAW (lower = sharper), Winkler Score  
**Propagation alert:** Precision, Recall, F1 (per pair and macro)

---

## Key Results

### Champion Model: SARIMA (PHX focus airport)

```
Champion model : SARIMA (1,0,2)(0,1,1)[7]
Champion RMSE  : 4.896 min
Beats best baseline by: 1.506 min RMSE (23.5% improvement)
```

### All-Airport RMSE Summary

| Airport | ARIMA | GRU | LSTM | QRNN | PICP | Winkler |
|---------|-------|-----|------|------|------|---------|
| ATL | 17.56 | 19.95 | 19.82 | 19.76 | 0.775 | 83.1 |
| DEN | 9.80 | 10.99 | 10.92 | 11.04 | 0.746 | 54.0 |
| DFW | 16.87 | 18.34 | 18.70 | 18.30 | 0.789 | 92.1 |
| ORD | 11.84 | 11.98 | 13.03 | 12.88 | 0.852 | 63.1 |
| CLT | 10.81 | 12.24 | 13.92 | 13.42 | 0.821 | 67.6 |
| LAX | 4.94 | 5.09 | 5.66 | 6.28 | 0.718 | 29.7 |
| PHX | **4.90** | 4.98 | 5.59 | 5.25 | **0.855** | **23.2** |
| LAS | 8.41 | 9.06 | 8.74 | 8.32 | 0.801 | 37.0 |
| SEA | 4.85 | 4.97 | 5.12 | 5.28 | 0.795 | 24.6 |
| MCO | 10.75 | 12.11 | 12.65 | 11.64 | 0.729 | 56.2 |

### Residual Noise (RQ1)

All 10 airports reject Gaussian at α = 0.05. Shapiro-Wilk p ranges from 3.2×10⁻¹⁶ (SEA) to 1.6×10⁻³⁰ (DFW). Log-Normal (shifted) wins AIC comparison (3600 vs Normal 3793).

### Propagation & Alerts (RQ3)

| Metric | Daily (old) | Hourly (v5) | Improvement |
|--------|------------|-------------|-------------|
| Macro Precision | 0.414 | **0.803** | +38.9 pp |
| Macro Recall | 0.403 | **0.730** | +32.7 pp |
| Macro F1 | 0.364 | **0.760** | +39.6 pp (×2.1) |

Top alert pair: **CLT→MCO** F1 = 0.892, Precision = 0.900  
Highest precision: **PHX→DFW** Precision = 0.930 (near-zero false alarms)  
Mean tail-chain carry-over: **74.2%** across 793,941 chain pairs

---

## Dashboard Guide

![Map](dashboard/dash1.png)
![Map](dashboard/dash2.png)


The Streamlit dashboard (`dash.py`) has 8 pages:

| Page | Description |
|------|-------------|
| 📊 Overview | Pipeline summary, key KPIs, PHX RMSE chart |
| 🚨 Ops Center | Live propagation risk map, active alerts, recommended actions |
| 🛬 Airport Explorer | Per-airport forecast chart, QRNN reliability diagram, model comparison |
| 📈 Model Comparison | PHX focus, master table (all airports), ARIMA vs GRU vs QRNN |
| 🔬 Noise Analysis | Residual distributions, normality tests, log-likelihood comparison |
| 🌐 Propagation | Granger causality heatmap, tail-number chains, top propagation pairs |
| 🕐 Hourly Alert System | Per-pair Precision/Recall/F1, daily vs hourly improvement |
| ❓ RQ Answers | Definitive answers to RQ1, RQ2, RQ3 with supporting charts |

**Ops Center alert mechanics:**
- Source signal: rolling 3-hour average of hourly mean departure delay
- Alert fires when source signal > configurable threshold (default: 15 min)
- Target window: Granger lag × 24h ± 6h slack
- Only 68 Granger-significant pairs (p < 0.05) are used

---

## Reproducibility

All random seeds are fixed globally:

```python
SEED = 42
import random, numpy as np, torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

Train/test split is **chronological** (no shuffling, no data leakage). All lags use `.shift(1)` to prevent look-ahead bias. Rolling statistics use `.shift(1).rolling(window)`.

---

## Dependencies

See `requirements.txt` for the full pinned list. Core libraries:

- **Data:** `pandas`, `numpy`, `pyarrow`
- **Statistics:** `statsmodels`, `pmdarima`, `scipy`
- **Deep Learning:** `torch`, `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Dashboard:** `streamlit`

---
