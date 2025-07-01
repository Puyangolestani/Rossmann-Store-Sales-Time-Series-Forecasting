# Rossmann-Store-Sales-Time-Series-Forecasting

# Final_Project_Rossmann

## Predicting Retail Sales with Time Series Forecasting

Welcome to the GitHub repository for my **Rossmann Store Sales Time Series Forecasting Project** — built during my data science learning journey. This project focuses on using **time series modeling** and machine learning to forecast daily sales for thousands of stores across Germany, enabling data-driven decision-making for promotions, staffing, and inventory.

---

## 🧠 Project Overview

**"Predicting Retail Sales with Time Series Forecasting"** leverages historical data from Rossmann drugstores to predict **future daily sales** over a 6-week horizon — the goal set by the [Rossmann Kaggle competition](https://www.kaggle.com/competitions/rossmann-store-sales).

The project combines classic time series techniques with machine learning (XGBoost), including smart feature engineering and time-aware validation.

---

## 🎯 Project Objectives

- 📆 Forecast daily store sales for the next 6 weeks.
- 🔍 Engineer time-based and lag-based features for temporal patterns.
- 📉 Evaluate model performance using time series cross-validation.
- 🧠 Train a robust XGBoost regressor model on the enriched data.
- 📊 Visualize splits, forecasts, and validation results clearly.

---

## 🔧 Technical Approach

### 1. Data Preparation & Indexing

- Parsed and cleaned 843,000+ daily sales records.
- Converted `Date` into datetime format and used it as the index.
- Grouped data by day to align with the forecasting granularity.

### 2. Feature Engineering

- **Calendar Features**: Extracted `dayofweek`, `month`, `year`, `quarter`, etc.
- **Lag Features**: Created `lag1` (sales from same day last year) to capture seasonality.
- **Promo & Holiday Encoding**: Retained and used `Promo`, `SchoolHoliday`, and `StateHoliday`.

### 3. Time Series Cross-Validation

Used `TimeSeriesSplit` to ensure training only includes past data — no future leakage:

```python
tss = TimeSeriesSplit(n_splits=5, test_size=42, gap=1)
