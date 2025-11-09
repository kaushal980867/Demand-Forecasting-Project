import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / yt)) * 100)

def metrics_frame(name, y_true, y_pred):
    return pd.DataFrame([{
        "Model": name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE %": mape(y_true, y_pred)
    }])

def ensure_artifacts():
    Path("artifacts").mkdir(exist_ok=True)

def plot_forecast(ts, y_pred, title, fname):
    ensure_artifacts()
    plt.figure(figsize=(12,5))
    plt.plot(ts.index, ts.values, label="Actual")
    plt.plot(y_pred.index, y_pred.values, label="Forecast")
    plt.title(title)
    plt.legend()
    out = Path("artifacts")/fname
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    return str(out)
