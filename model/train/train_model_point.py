# train_model_point.py
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----- LOAD DATA -----
df = pd.read_csv("../data/synthetic_invoices_with_macro.csv", parse_dates=['invoice_date', 'payment_date'])

feature_cols = [
    'client_id',
    'invoice_amount',
    'home_sales',
    'avg_price',
    'overnight_rate',
    'invoice_year',
    'invoice_month',
]

X = df[feature_cols]
y = df['days_to_pay']

numeric_features = [
    'invoice_amount',
    'home_sales',
    'avg_price',
    'overnight_rate',
    'invoice_year',
    'invoice_month',
]

categorical_features = ['client_id']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ]
)

# ----- MODEL PIPELINE -----
regressor = GradientBoostingRegressor(random_state=42)

model = Pipeline(
    steps=[
        ('preprocess', preprocessor),
        ('regressor', regressor),
    ]
)

# ----- TRAIN/TEST SPLIT (for *real* data you should do time-based split) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"MAE: {mae:.2f} days, RMSE: {rmse:.2f} days")

# ----- SAVE MODEL -----
joblib.dump(model, "invoice_delay_point_model.joblib")
print("Saved invoice_delay_point_model.joblib")
