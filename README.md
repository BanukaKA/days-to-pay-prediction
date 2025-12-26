# Invoice Payment Date Forecasting API

Forecast when an invoice will be paid, given basic invoice details and macroeconomic context.

This project trains a machine learning model to predict **days-to-pay** for invoices and exposes it through a **FastAPI** endpoint. It uses a mix of **synthetic invoice data** and **real public macro data** (Ontario housing market stats and Canadian interest rates) to simulate realistic payment behaviour before plugging in real company data.

---

## Features

- **Synthetic invoice dataset generator**
  - Simulates clients, invoices, and payment dates.
  - Incorporates client segments (good / average / slow payers).
  - Adds realistic effects from:
    - Invoice amount  
    - Seasonality (e.g., slower payments in December/January)  
    - Housing market activity and interest rates  

- **Machine learning models**
  - Trains **CatBoostRegressor** models for:
    - Point prediction (minimizing MAE).
    - Quantile regression (P10 / P90) for prediction intervals.
  - Time-aware train/validation split (train on earlier invoices, validate on later ones).

- **FastAPI inference service**
  - `/predict` endpoint that accepts:
    - `client_id`
    - `invoice_amount`
    - `submission_date`
    - `home_sales`, `avg_price`, `overnight_rate`
  - Returns:
    - Predicted **payment date**
    - Lower and upper bound dates (P10–P90)
    - Corresponding days-to-pay values

- **Easy to swap synthetic data for real invoices**
  - Same schema and training flow can be applied to real historical invoice data from your accounting system.

---

## Tech Stack

- **Language:** Python
- **ML / Data:** CatBoost, scikit-learn, pandas, NumPy
- **API:** FastAPI, Uvicorn
- **Model persistence:** joblib

---

## Project Structure

```text
.
├── generate_synthetic_data.py        # Build synthetic invoice + macro dataset
├── train_catboost_models.py          # Train point + quantile CatBoost models
├── synthetic_invoices_with_macro.csv # Generated sample data (optional, .gitignored if large)
├── invoice_delay_catboost_models.joblib  # Saved model bundle
├── main.py                           # FastAPI app exposing /predict
└── README.md
