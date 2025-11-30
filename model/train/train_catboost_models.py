# train_catboost_models.py

import joblib
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

# ---------- 1. LOAD AND PREPARE DATA ----------

DATA_PATH = "../data/synthetic_invoices_with_macro.csv"  # change to your real file later

df = pd.read_csv(
    DATA_PATH,
    parse_dates=['invoice_date', 'payment_date']
)

# Sort by time so we can simulate "train on past, test on future"
df = df.sort_values('invoice_date').reset_index(drop=True)

# Features and target
feature_cols = [
    'client_id',
    'invoice_amount',
    'home_sales',
    'avg_price',
    'overnight_rate',
    'invoice_year',
    'invoice_month',
]
target_col = 'days_to_pay'

X = df[feature_cols]
y = df[target_col]

# 80% earliest invoices for train, 20% latest for validation
split_idx = int(0.8 * len(df))

X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

# Tell CatBoost which features are categorical (we only have client_id)
cat_features = ['client_id']

train_pool = Pool(
    X_train,
    y_train,
    cat_features=cat_features
)

val_pool = Pool(
    X_val,
    y_val,
    cat_features=cat_features
)

print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

# ---------- 2. TRAIN POINT MODEL (MAE) ----------

point_model = CatBoostRegressor(
    loss_function='MAE',     # minimize absolute days error
    eval_metric='MAE',
    depth=8,                 # tree depth
    learning_rate=0.05,
    iterations=2000,         # max trees
    l2_leaf_reg=3.0,         # regularization
    random_state=42,
    od_type='Iter',          # early stopping
    od_wait=100,             # stop if no improvement for 100 iters
    verbose=100,
)

point_model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

# Evaluate on validation
val_pred_point = point_model.predict(val_pool)
val_mae_point = mean_absolute_error(y_val, val_pred_point)
print(f"[POINT MODEL] Validation MAE: {val_mae_point:.2f} days")

# ---------- 3. TRAIN QUANTILE MODELS (P10 and P90) ----------

def train_quantile(alpha: float, train_pool: Pool, val_pool: Pool):
    """
    Train a CatBoost quantile model for a given alpha.
    """
    model = CatBoostRegressor(
        loss_function=f'Quantile:alpha={alpha}',
        eval_metric='Quantile:alpha={}'.format(alpha),
        depth=8,
        learning_rate=0.05,
        iterations=2000,
        l2_leaf_reg=3.0,
        random_state=42,
        od_type='Iter',
        od_wait=100,
        verbose=100,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model

lower_model = train_quantile(alpha=0.10, train_pool=train_pool, val_pool=val_pool)
upper_model = train_quantile(alpha=0.90, train_pool=train_pool, val_pool=val_pool)

# Quick sanity check: median of lower/upper errors (not super meaningful, but gives a feel)
val_pred_lower = lower_model.predict(val_pool)
val_pred_upper = upper_model.predict(val_pool)

val_mae_lower = mean_absolute_error(y_val, val_pred_lower)
val_mae_upper = mean_absolute_error(y_val, val_pred_upper)
print(f"[LOWER MODEL] Validation MAE vs true days: {val_mae_lower:.2f} days")
print(f"[UPPER MODEL] Validation MAE vs true days: {val_mae_upper:.2f} days")

# ---------- 4. SAVE MODELS BUNDLE ----------

bundle = {
    "point": point_model,
    "lower": lower_model,
    "upper": upper_model,
    "feature_cols": feature_cols,
}

joblib.dump(bundle, "invoice_delay_catboost_models.joblib")
print("Saved invoice_delay_catboost_models.joblib")
