import pandas as pd
from statsmodels.tsa.stattools import adfuller

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = ["Date", "Sales"]

    # Remove rows where Date or Sales is missing
    df = df.dropna(subset=["Date", "Sales"], how='any')

    # Remove the row containing text instead of date (row 106 in original dataset)
    df = df[df["Date"].str.contains(r"^\d{4}-\d{2}")]
    
    # Convert to datetime safely
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")

    # Convert Sales to numeric (just in case)
    df["Sales"] = pd.to_numeric(df["Sales"], errors='coerce')

    # Drop any remaining NaN from conversion
    df = df.dropna()

    # Set index
    df.set_index("Date", inplace=True)
    df = df.sort_index()

    # Ensure monthly frequency (important for ARIMA/SARIMA)
    df.index.freq = "MS"

    return df["Sales"]


def seasonal_first_difference(ts, season=12):
    return ts - ts.shift(season)

def adf_test(series, title=''):
    print(f'\nADF Test: {title}')
    result = adfuller(series.dropna())
    labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
    out = pd.Series(result[0:4], index=labels)
    for key, value in result[4].items():
        out[f'Critical Value ({key})'] = value
    print(out.to_string())
    if result[1] <= 0.05:
        print("=> Strong evidence against H0 — Data is STATIONARY")
    else:
        print("=> Weak evidence against H0 — Data is NON-STATIONARY")


def adf_summary(series):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]

    if p_value <= 0.05:
        conclusion = "Data is Stationary ✅ (Good for forecasting)"
    else:
        conclusion = "Data is Non-Stationary ⚠️ (Needs differencing)"

    return adf_stat, p_value, conclusion
