# generate_synthetic_data.py
import numpy as np
import pandas as pd

# ------------- CONFIG -------------
N_INVOICES = 10000
N_CLIENTS = 80
RNG_SEED = 42

np.random.seed(RNG_SEED)

# ------------- LOAD MACRO DATA -------------

# 1) Ontario monthly housing data
# Adapt column names to whatever your download actually has.
housing = pd.read_csv("ontario_housing_monthly.csv")
housing['date'] = pd.to_datetime(housing['date'])
housing['year_month'] = housing['date'].dt.to_period('M')

# Keep only what we need
housing = housing[['year_month', 'home_sales', 'avg_price']]

# 2) Interest rate data
rates = pd.read_csv("canada_interest_rate_monthly.csv")
rates['date'] = pd.to_datetime(rates['date'])
rates['year_month'] = rates['date'].dt.to_period('M')
rates = rates[['year_month', 'overnight_rate']]

# 3) Merge into a single macro monthly table
macro = housing.merge(rates, on='year_month', how='inner').sort_values('year_month')

# Use continuous timestamp for sampling dates later
macro['month_start'] = macro['year_month'].dt.to_timestamp()

print("Macro rows:", len(macro), macro.head())

# ------------- SYNTHETIC CLIENTS -------------

clients = [f"C{i:03d}" for i in range(1, N_CLIENTS + 1)]

# Assign each client a “segment” that drives payment speed & variability
segments = np.random.choice(['A', 'B', 'C'], size=N_CLIENTS, p=[0.3, 0.5, 0.2])

segment_base_delay = {'A': 25, 'B': 40, 'C': 60}   # typical days to pay
segment_volatility = {'A': 5,  'B': 10, 'C': 15}   # std dev of noise

client_df = pd.DataFrame({
    'client_id': clients,
    'segment': segments
})
client_df['base_delay'] = client_df['segment'].map(segment_base_delay)
client_df['volatility'] = client_df['segment'].map(segment_volatility)

print(client_df.head())

# ------------- GENERATE INVOICES -------------

# Helper: sample random invoice months
macro_idx = macro.index.to_numpy()

rows = []
for _ in range(N_INVOICES):
    # Pick a client
    client = client_df.sample(1).iloc[0]
    
    # Pick a random month from macro data
    mrow = macro.loc[np.random.choice(macro_idx)]
    
    # Random day in that month (approx 30 days)
    day_offset = np.random.randint(0, 28)
    invoice_date = mrow['month_start'] + pd.Timedelta(days=int(day_offset))
    
    # Invoice amount: log-normalish, slightly higher for slower-paying segments
    base_amount = np.random.lognormal(mean=11, sigma=0.5)  # roughly around ~60k but skewed
    if client['segment'] == 'C':
        base_amount *= 1.3
    elif client['segment'] == 'A':
        base_amount *= 0.8
    invoice_amount = np.round(base_amount, 2)
    
    # Macro features
    home_sales = mrow['home_sales']
    avg_price = mrow['avg_price']
    overnight_rate = mrow['overnight_rate']
    
    # ---------- SYNTHETIC PAYMENT DELAY MODEL ----------
    # Base client delay
    mu = client['base_delay']
    
    # Higher interest rates -> slower payments
    # Center interest rate around its mean so effect is relative
    rate_centered = overnight_rate - macro['overnight_rate'].mean()
    mu += 4.0 * rate_centered  # each 1% above mean adds ~4 days
    
    # Weaker home sales might mean slower cash flow
    home_sales_centered = home_sales - macro['home_sales'].mean()
    mu -= 0.0003 * home_sales_centered  # more sales => slightly faster
    
    # Larger invoices tend to be paid later
    mu += 0.005 * (invoice_amount / 1000.0)  # per $1k
    
    # Seasonality: December is slower (holidays), Jan slightly slower
    month = invoice_date.month
    if month == 12:
        mu += 7
    elif month == 1:
        mu += 3
    
    # Add client-specific noise
    noise = np.random.normal(loc=0, scale=client['volatility'])
    days_to_pay = int(max(5, mu + noise))  # at least 5 days
    
    payment_date = invoice_date + pd.Timedelta(days=days_to_pay)
    
    rows.append({
        'client_id': client['client_id'],
        'segment': client['segment'],
        'invoice_amount': invoice_amount,
        'invoice_date': invoice_date,
        'home_sales': home_sales,
        'avg_price': avg_price,
        'overnight_rate': overnight_rate,
        'days_to_pay': days_to_pay,
        'payment_date': payment_date
    })

df = pd.DataFrame(rows)

# Some extra time features (useful for the model)
df['invoice_year'] = df['invoice_date'].dt.year
df['invoice_month'] = df['invoice_date'].dt.month

df.to_csv("synthetic_invoices_with_macro.csv", index=False)
print("Saved synthetic_invoices_with_macro.csv with", len(df), "rows")
