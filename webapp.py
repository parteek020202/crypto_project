#!/usr/bin/env python
# coding: utf-8

# In[42]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
import matplotlib.pyplot as plt
import datetime

# Define list of crypto tickers
tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]

# Sidebar for user inputs
st.sidebar.title("Crypto Price Prediction Dashboard")
ticker = st.sidebar.selectbox("Select Crypto", tickers)
days_back = st.sidebar.slider("Historical Data Period (Days)", 60, 365, 180)
prediction_horizon = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=30, value=1)

# Determine filenames based on ticker (using joblib for loading)

model_filename = f"{ticker}_svr_model.pkl"
scaler_filename = f"{ticker}_scaler_X_svr.pkl"

try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
except Exception as e:
    st.error(f"Error loading model or scaler for {ticker}: {e}")
    st.stop()

# Fetch historical data using yfinance
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=days_back)
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("No data fetched. Please check the ticker or date range.")
    st.stop()

# Compute additional features
data['Lag1'] = data['Close'].shift(1)
data['Lag7'] = data['Close'].shift(7)
data['SMA7'] = data['Close'].rolling(window=7).mean()


# Ensure that 'Close' is a 1D Series
close_series = data['Close'].squeeze()

# Calculate RSI14 using the ta library
rsi = RSIIndicator(close=close_series, window=14)
data['RSI14'] = rsi.rsi()

# Calculate MACD using the ta library
macd = MACD(close=close_series)
data['MACD'] = macd.macd()

# Daily returns and volatility
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=7).std()

# Drop NA values from shifts and rolling computations
data = data.dropna()

# Define feature list and create target variable (next day closing price)
features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7",
            "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Prepare the latest available data for prediction
latest_features = data[features].iloc[-1].values.reshape(1, -1)
latest_features_scaled = scaler.transform(latest_features)
predicted_price = model.predict(latest_features_scaled)[0]

# Display prediction results
st.header(f"{ticker} Price Prediction")
st.write(f"**Predicted next day closing price:** ${predicted_price:.2f}")

# Generate a simple buy/sell signal: if predicted price > current close, signal "Buy"
# Ensure predicted_price and current_close are scalars
predicted_price = float(predicted_price)
current_close = float(data['Close'].iloc[-1])

signal = "Buy" if predicted_price > current_close else "Sell"

st.subheader(f"Trading Signal: {signal}")

# Plot historical closing prices with predicted price marker
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data['Close'], label="Historical Close")
# Place the predicted price one day after the last available date
ax.scatter(data.index[-1] + pd.Timedelta(days=1), predicted_price, color='red', label="Predicted Price")
ax.set_title(f"{ticker} Price History and Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Show today's market data (intraday)
st.header("Today's Market Data")
today_data = yf.download(ticker, period="1d", interval="1m")
if not today_data.empty:
    st.line_chart(today_data['Close'])
else:
    st.write("Intraday data not available.")

# Display technical indicators
st.header("Technical Indicators")
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# RSI plot
ax1.plot(data.index, data['RSI14'], label="RSI14", color='purple')
ax1.axhline(70, color='red', linestyle='--')
ax1.axhline(30, color='green', linestyle='--')
ax1.set_ylabel("RSI")
ax1.legend()

# MACD plot
ax2.plot(data.index, data['MACD'], label="MACD", color='blue')
ax2.set_ylabel("MACD")
ax2.legend()
st.pyplot(fig2)

# Additional Dashboard Information
st.markdown("### Additional Dashboard Features")
st.markdown("- Detailed historical prediction analysis")
st.markdown("- Interactive date range filtering")
st.markdown("- Customizable technical indicator thresholds")
st.markdown("- Comparison charts: Predicted vs. Actual Prices")

# Future expansion: You can add more interactive elements, notifications,
# and even integrate external APIs for news or sentiment analysis.


# In[ ]:




