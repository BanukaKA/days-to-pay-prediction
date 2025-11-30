# train_model_quantiles.py
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def make_pipeline(loss, alpha=None):
    kwargs = {'random_state': 42, 'loss': loss}
    if alpha is not None:
        kwargs['alpha'] = alpha
    reg = GradientBoostingRegressor(**kwargs)
    return Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('regressor', reg),
        ]
    )

median_model = make_pipeline(loss='squared_error')         # ~mean/median
lower_model  = make_pipeline(loss='quantile', alpha=0.1)   # 10th percentile
upper_model  = make_pipeline(loss='quantile', alpha=0.9)   # 90th percentile

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

median_model.fit(X_train, y_train)
lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

# Evaluate central model
y_pred = median_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"[Median model] MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Save all 3
bundle = {
    "median": median_model,
    "lower": lower_model,
    "upper": upper_model,
    "feature_cols": feature_cols,
}
joblib.dump(bundle, "invoice_delay_quantile_models.joblib")
print("Saved invoice_delay_quantile_models.joblib")
