# main.py
from datetime import date, timedelta
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ----- LOAD MODELS AT STARTUP -----
models_bundle = joblib.load("invoice_delay_catboost_models.joblib")
point_model = models_bundle["point"]
lower_model = models_bundle["lower"]
upper_model = models_bundle["upper"]
feature_cols = models_bundle["feature_cols"]

app = FastAPI(title="Invoice Payment Date Forecaster")

# ----- SCHEMAS -----

class InvoiceInput(BaseModel):
    client_id: str
    invoice_amount: float
    submission_date: date
    home_sales: float          # monthly home sales at time of invoicing
    avg_price: float           # average home price
    overnight_rate: float      # e.g. BoC overnight rate as %
    # If you prefer, you can pass invoice_year/month directly instead of deriving.

class PredictionOutput(BaseModel):
    predicted_payment_date: date
    lower_payment_date: date
    upper_payment_date: date
    predicted_days_to_pay: float
    lower_days_to_pay: float
    upper_days_to_pay: float

# ----- HELPER -----

def build_feature_frame(payload: InvoiceInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "client_id": payload.client_id,
        "invoice_amount": payload.invoice_amount,
        "home_sales": payload.home_sales,
        "avg_price": payload.avg_price,
        "overnight_rate": payload.overnight_rate,
        "invoice_year": payload.submission_date.year,
        "invoice_month": payload.submission_date.month,
    }])

# ----- ROUTES -----

@app.get("/")
def root():
    return {"message": "Invoice payment date forecasting API. POST to /predict"}

@app.post("/predict", response_model=PredictionOutput)
def predict_payment_date(invoice: InvoiceInput):
    X = build_feature_frame(invoice)

    days_med = float(point_model.predict(X)[0])
    days_lo  = float(lower_model.predict(X)[0])
    days_hi  = float(upper_model.predict(X)[0])

    # Ensure order (quantile models should already respect this, but just in case)
    lower_days = min(days_lo, days_med, days_hi)
    upper_days = max(days_lo, days_med, days_hi)
    pred_days  = days_med

    # Convert to dates (round to nearest day)
    pred_date  = invoice.submission_date + timedelta(days=round(pred_days))
    lower_date = invoice.submission_date + timedelta(days=round(lower_days))
    upper_date = invoice.submission_date + timedelta(days=round(upper_days))

    return PredictionOutput(
        predicted_payment_date=pred_date,
        lower_payment_date=lower_date,
        upper_payment_date=upper_date,
        predicted_days_to_pay=pred_days,
        lower_days_to_pay=lower_days,
        upper_days_to_pay=upper_days,
    )
