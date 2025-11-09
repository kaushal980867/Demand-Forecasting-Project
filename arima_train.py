import pandas as pd
from statsmodels.tsa.arima.model import ARIMA  # modern import

def fit_and_forecast(ts: pd.Series,
                     order=(1,1,1),
                     start_idx: int | None = 90,
                     end_idx: int | None = 103,
                     dynamic: bool = True,
                     true_split: bool = False,
                     test_points: int = 14):
    """
    If true_split=False (default) -> mimic your notebook: fit on full series, predict start..end.
    If true_split=True -> fit on train ([:-test_points]), forecast next test_points.
    """
    if not true_split:
        model = ARIMA(ts, order=order)
        fit = model.fit()
        # same as your code:
        preds = fit.predict(start=start_idx, end=end_idx, dynamic=dynamic)
        preds.index = ts.index[start_idx:end_idx+1]
        return fit, preds
    else:
        train = ts.iloc[:-test_points]
        test = ts.iloc[-test_points:]
        model = ARIMA(train, order=order)
        fit = model.fit()
        fc = fit.forecast(steps=test_points)
        fc.index = test.index
        return fit, fc
