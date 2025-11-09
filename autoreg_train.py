import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

def fit_and_forecast(ts: pd.Series, lags: int = 12, test_points: int = 14):
    train = ts.iloc[:-test_points]
    test = ts.iloc[-test_points:]
    model = AutoReg(train, lags=lags, old_names=False)
    res = model.fit()
    fc = res.predict(start=len(train), end=len(train)+test_points-1, dynamic=False)
    fc.index = test.index
    return res, fc
