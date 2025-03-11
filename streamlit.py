#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates
from datetime import datetime

# Title and sidebar
st.title("Advanced Crypto Price Prediction Dashboard")
st.sidebar.header("Settings")
ticker = st.sidebar.selectbox("Select Cryptocurrency", ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"])
lookback = st.sidebar.slider("Lookback Period (days)", 7, 30, 14)

# Load data and model
@st.cache_data
def load_data(ticker):
    df = pd.read_csv(f"{ticker}_processed.csv", index_col="Date", parse_dates=True)
    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
    X = df[features].values
    y = df["Target"].values
    return df, X, y, features

@st.cache_resource
def load_model(ticker):
    model = joblib.load(f"{ticker}_xgboost_model.pkl")
    scaler_X = joblib.load(f"{ticker}_scaler_X_xgb.pkl")
    return model, scaler_X

df, X, y, features = load_data(ticker)
model, scaler_X = load_model(ticker)

# Prepare data for prediction
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Train or load model (simplified for demo)
model.fit(X_train_scaled, y_train)
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
signals.loc[(signals["Predicted"] > signals["Actual"] * (1 + threshold)), "Signal"] = 1
signals.loc[(signals["Predicted"] < signals["Actual"] * (1 - threshold)), "Signal"] = -1

# Visualizations
st.header(f"{ticker} Predictions and Signals")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(signals.index, signals["Actual"], label="Actual Price", color="blue")
ax1.plot(signals.index, signals["Predicted"], label="Predicted Price", color="orange", linestyle="--")
ax1.scatter(signals[signals["Signal"] == 1].index, signals[signals["Signal"] == 1]["Actual"], 
            marker="^", color="green", label="Buy", s=100)
ax1.scatter(signals[signals["Signal"] == -1].index, signals[signals["Signal"] == -1]["Actual"], 
            marker="v", color="red", label="Sell", s=100)
ax1.set_title(f"{ticker} Price Prediction and Signals")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
ax1.legend()
ax1.grid(True)
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig1)

# Feature Importance
st.header("Feature Importance")
importance = model.feature_importances_
feat_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feat_importance = feat_importance.sort_values('Importance', ascending=False)
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(feat_importance['Feature'], feat_importance['Importance'])
ax2.set_title(f"{ticker} Feature Importance")
ax2.set_xlabel("Features")
ax2.set_ylabel("Importance")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig2)

# Metrics
st.header("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
st.write(f"Directional Accuracy: {da:.2f}%")  # Keep it minimal, as per your request

# Interactive Signal Filter
st.header("Signal Filter")
signal_type = st.selectbox("Show Signals", ["All", "Buy Only", "Sell Only"])
if signal_type == "Buy Only":
    signals_to_show = signals[signals["Signal"] == 1]
elif signal_type == "Sell Only":
    signals_to_show = signals[signals["Signal"] == -1]
else:
    signals_to_show = signals
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(signals_to_show.index, signals_to_show["Actual"], label="Actual Price", color="blue")
ax3.plot(signals_to_show.index, signals_to_show["Predicted"], label="Predicted Price", color="orange", linestyle="--")
if not signals_to_show.empty:
    ax3.scatter(signals_to_show[signals_to_show["Signal"] == 1].index, signals_to_show[signals_to_show["Signal"] == 1]["Actual"], 
                marker="^", color="green", label="Buy", s=100)
    ax3.scatter(signals_to_show[signals_to_show["Signal"] == -1].index, signals_to_show[signals_to_show["Signal"] == -1]["Actual"], 
                marker="v", color="red", label="Sell", s=100)
ax3.set_title(f"{ticker} Filtered Signals")
ax3.set_xlabel("Date")
ax3.set_ylabel("Price (USD)")
ax3.legend()
ax3.grid(True)
ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
st.pyplot(fig3)

# Downloadable Report
st.header("Download Report")
report_data = signals[["Actual", "Predicted", "Signal"]].to_csv(index=True)
st.download_button(label="Download Predictions", data=report_data, file_name=f"{ticker}_predictions.csv", mime="text/csv")

# Footer
st.sidebar.text("Built with Streamlit by Grok 3 (xAI) and You!")

# Run with: streamlit run crypto_dashboard.py


# In[ ]:




