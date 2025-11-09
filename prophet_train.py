import pandas as pd

def _to_prophet_df(ts: pd.Series) -> pd.DataFrame:
    dfp = pd.DataFrame({"ds": ts.index, "y": ts.values})
    return dfp

def fit_and_forecast(ts: pd.Series, test_points: int = 14):
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet  # fallback if needed

    train = ts.iloc[:-test_points]
    test = ts.iloc[-test_points:]

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(_to_prophet_df(train))
    future = m.make_future_dataframe(periods=test_points, freq="MS")
    fc = m.predict(future)
    fc_tail = fc.tail(test_points).set_index("ds")["yhat"]
    fc_tail.index = test.index
    return m, fc_tail
