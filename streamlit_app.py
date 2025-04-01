#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[6]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates

# Title and sidebar
st.title("Advanced Crypto Price Prediction Dashboard")
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Cryptocurrency", ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"])

# Load data and model
@st.cache_data(hash_funcs={"builtins.str": hash})
def load_data(ticker):
    df = pd.read_csv(f"{ticker}_processed.csv", index_col="Date", parse_dates=True)
    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
    X = df[features].values
    y = df["Target"].values
    return df, X, y

@st.cache_resource(hash_funcs={"builtins.str": hash})
def load_model(ticker):
    model = joblib.load(f"{ticker}_svr_model.pkl")
    scaler_X = joblib.load(f"{ticker}_scaler_X_svr.pkl")
    return model, scaler_X

df, X, y = load_data(ticker)
model, scaler_X = load_model(ticker)

# Prepare data for prediction
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Predictions
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
test_dates = df.index[train_size:]
actual_series = pd.Series(y_test, index=test_dates)
pred_series = pd.Series(y_pred, index=test_dates)
actual_direction = (actual_series.shift(-1) > actual_series).iloc[:-1].astype(int)
predicted_direction = (pred_series.shift(-1) > pred_series).iloc[:-1].astype(int)
da = (actual_direction == predicted_direction).mean() * 100

# Signal generation
threshold = 0.015
signals = pd.DataFrame(index=test_dates)
signals["Actual"] = actual_series
signals["Predicted"] = pred_series
signals["Signal"] = 0
signals.loc[signals["Predicted"] > signals["Actual"] * (1 + threshold), "Signal"] = 1
signals.loc[signals["Predicted"] < signals["Actual"] * (1 - threshold), "Signal"] = -1

# Visualizations
st.header(f"{ticker} Predictions and Signals")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(signals.index, signals["Actual"], label="Actual Price", color="blue")
ax.plot(signals.index, signals["Predicted"], label="Predicted Price", color="orange", linestyle="--")
ax.scatter(signals[signals["Signal"] == 1].index, signals[signals["Signal"] == 1]["Actual"], marker="^", color="green", label="Buy", s=100)
ax.scatter(signals[signals["Signal"] == -1].index, signals[signals["Signal"] == -1]["Actual"], marker="v", color="red", label="Sell", s=100)
ax.set_title(f"{ticker} Price Prediction and Signals")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid(True)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig)

# Metrics
st.header("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
st.write(f"Directional Accuracy: {da:.2f}%")

# Downloadable Report
st.header("Download Report")
report_data = signals.to_csv(index=True)
st.download_button(label="Download Predictions", data=report_data, file_name=f"{ticker}_predictions.csv", mime="text/csv")


# Run with streamlit1.py


# In[ ]:




