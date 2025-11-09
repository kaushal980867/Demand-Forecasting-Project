import streamlit as st
import pandas as pd
from pathlib import Path

from data_prep import load_and_clean
from utils.eval import metrics_frame, plot_forecast
from models.arima_train import fit_and_forecast as arima_ff
from models.sarimax_train import fit_and_forecast as sarimax_ff
from models.prophet_train import fit_and_forecast as prophet_ff
from models.autoreg_train import fit_and_forecast as autoreg_ff

st.set_page_config(page_title="Champagne Sales: Model Comparison", layout="wide")
st.title("🍾 Champagne Sales Forecast — Model Comparison")

# --- Data input
uploaded = st.file_uploader("Upload the dataset CSV (or use bundled one)", type=["csv"])
if uploaded is not None:
    csv_path = uploaded
else:
    csv_path = Path("data/perrin-freres-monthly-champagne-.csv")
    st.caption("Using bundled dataset.")

ts = load_and_clean(csv_path)

st.subheader("Data Preview")
st.line_chart(ts)

from data_prep import adf_summary

st.subheader("Stationarity Check (ADF Test)")
adf_stat, p_val, conclusion = adf_summary(ts)
st.write(f"**ADF Statistic:** {adf_stat:.4f}")
st.write(f"**p-value:** {p_val:.6f}")
st.write(f"**Conclusion:** {conclusion}")


with st.sidebar:
    st.header("Settings")
    test_points = st.number_input("Test points (last N months)", min_value=6, max_value=36, value=14, step=1)
    mimic_notebook = st.checkbox("Mimic notebook logic for ARIMA/SARIMAX (fit on full, predict start..end)", value=True)
    start_idx = st.number_input("Start index (for mimic mode)", min_value=0, max_value=len(ts)-1, value=90)
    end_idx = st.number_input("End index (for mimic mode)", min_value=0, max_value=len(ts)-1, value=103)

    models_to_run = st.multiselect(
        "Models",
        ["ARIMA(1,1,1)", "SARIMAX(1,1,1)x(1,1,1,12)", "Prophet", "AutoReg(lags=12)"],
        default=["ARIMA(1,1,1)", "SARIMAX(1,1,1)x(1,1,1,12)", "Prophet", "AutoReg(lags=12)"]
    )

st.subheader("Train & Compare")

rows = []
plots = []

def add_result(name, truth, preds):
    rows.append(metrics_frame(name, truth, preds))
    plot_path = plot_forecast(ts, preds, f"{name} Forecast", f"{name.replace(' ','_')}.png")
    plots.append((name, plot_path))

# Targets for evaluation
test = ts.iloc[-test_points:]

# Run selected models
if "ARIMA(1,1,1)" in models_to_run:
    if mimic_notebook:
        _, preds = arima_ff(ts, order=(1,1,1), start_idx=int(start_idx), end_idx=int(end_idx), dynamic=True, true_split=False)
        truth = ts.loc[preds.index]
    else:
        _, preds = arima_ff(ts, order=(1,1,1), true_split=True, test_points=test_points)
        truth = test
    add_result("ARIMA(1,1,1)", truth, preds)

if "SARIMAX(1,1,1)x(1,1,1,12)" in models_to_run:
    if mimic_notebook:
        _, preds = sarimax_ff(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                              start_idx=int(start_idx), end_idx=int(end_idx), dynamic=True, true_split=False)
        truth = ts.loc[preds.index]
    else:
        _, preds = sarimax_ff(ts, order=(1,1,1), seasonal_order=(1,1,1,12), true_split=True, test_points=test_points)
        truth = test
    add_result("SARIMAX(1,1,1)x(1,1,1,12)", truth, preds)

if "Prophet" in models_to_run:
    try:
        _, preds = prophet_ff(ts, test_points=test_points)
        add_result("Prophet", test, preds)
    except Exception as e:
        st.warning(f"Prophet failed to run: {e}")

if "AutoReg(lags=12)" in models_to_run:
    _, preds = autoreg_ff(ts, lags=12, test_points=test_points)
    add_result("AutoReg(lags=12)", test, preds)

if rows:
    comp = pd.concat(rows, ignore_index=True).sort_values("RMSE")
    st.write("### Metrics (lower is better)")
    st.dataframe(comp, use_container_width=True)

    st.write("### Forecast Plots")
    cols = st.columns(2)
    for i, (name, pth) in enumerate(plots):
        with cols[i % 2]:
            st.image(pth, caption=name)
