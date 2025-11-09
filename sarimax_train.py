import pandas as pd
import statsmodels.api as sm

def fit_and_forecast(ts: pd.Series,
                     order=(1,1,1),
                     seasonal_order=(1,1,1,12),
                     start_idx: int | None = 90,
                     end_idx: int | None = 103,
                     dynamic: bool = True,
                     true_split: bool = False,
                     test_points: int = 14):
    if not true_split:
        model = sm.tsa.statespace.SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                                          enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        preds = res.predict(start=start_idx, end=end_idx, dynamic=dynamic)
        preds.index = ts.index[start_idx:end_idx+1]
        return res, preds
    else:
        train = ts.iloc[:-test_points]
        test = ts.iloc[-test_points:]
        model = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                          enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.forecast(steps=test_points)
        fc.index = test.index
        return res, fc
