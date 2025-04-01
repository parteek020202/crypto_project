#!/usr/bin/env python
# coding: utf-8

# In[44]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
import plotly.graph_objects as go
import datetime

# Title and description
st.title("Advanced Crypto Price Prediction & Dashboard")
st.markdown("""
This interactive dashboard provides predictions, trading signals, historical analysis, and technical indicators for your favorite cryptocurrencies.
""")

# Sidebar for user inputs
st.sidebar.header("Settings")
tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
ticker = st.sidebar.selectbox("Select Crypto", tickers)
days_back = st.sidebar.slider("Historical Data Period (Days)", 60, 730, 180)
prediction_horizon = st.sidebar.number_input("Days to Forecast", min_value=1, max_value=30, value=1)

# Determine filenames based on ticker (using joblib for loading)

model_filename = f"{ticker}_svr_model.pkl"
scaler_filename = f"{ticker}_scaler_X_svr.pkl"

# Load model and scaler
try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
except Exception as e:
    st.error(f"Error loading model or scaler for {ticker}: {e}")
    st.stop()

# Fetch historical data
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=days_back)
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("No data fetched. Please check the ticker or date range.")
    st.stop()

# Feature Engineering
data['Lag1'] = data['Close'].shift(1)
data['Lag7'] = data['Close'].shift(7)
data['SMA7'] = data['Close'].rolling(window=7).mean()

# Calculate RSI14 using ta library (ensuring Close is a Series)
close_series = pd.Series(np.squeeze(data['Close'].to_numpy()), index=data.index)
rsi = RSIIndicator(close=close_series, window=14)
data['RSI14'] = rsi.rsi()

# Calculate MACD using ta library
macd = MACD(close=close_series)
data['MACD'] = macd.macd()

# Daily returns and volatility
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=7).std()

# Drop missing values
data = data.dropna()

# Define features and target
features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7",
            "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Prepare latest available feature for prediction
latest_features = data[features].iloc[-1].values.reshape(1, -1)
latest_features_scaled = scaler.transform(latest_features)
predicted_price = model.predict(latest_features_scaled)[0]

# Convert to float for comparison
predicted_price = float(predicted_price)
current_close = float(data['Close'].iloc[-1])
signal = "Buy" if predicted_price > current_close else "Sell"

# Create Tabs for different views
tabs = st.tabs(["Dashboard", "Technical Indicators", "Historical Analysis", "Prediction History"])

# Dashboard Tab
with tabs[0]:
    st.header(f"{ticker} - Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Close", f"${current_close:.2f}")
    col2.metric("Predicted Next Close", f"${predicted_price:.2f}")
    col3.metric("Trading Signal", signal)
    
    # Display a Plotly chart for historical closing prices with predicted marker
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
    fig.add_trace(go.Scatter(x=[data.index[-1] + pd.Timedelta(days=1)], y=[predicted_price],
                             mode='markers', marker=dict(color='red', size=10), name='Predicted Price'))
    fig.update_layout(title=f"{ticker} Price History with Prediction", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Today's Market Data")
    today_data = yf.download(ticker, period="1d", interval="1m")
    if not today_data.empty:
        st.line_chart(today_data['Close'])
    else:
        st.write("Intraday data not available.")

# Technical Indicators Tab
with tabs[1]:
    st.header("Technical Indicators")
    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI14'], mode='lines', name='RSI14'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="RSI (14) over Time", yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    fig_macd.update_layout(title="MACD over Time", yaxis_title="MACD")
    st.plotly_chart(fig_macd, use_container_width=True)

# Historical Analysis Tab
with tabs[2]:
    st.header("Historical Data & Analysis")
    date_range = st.date_input("Select Date Range", [data.index.min(), data.index.max()])
    if len(date_range) == 2:
        mask = (data.index.date >= date_range[0]) & (data.index.date <= date_range[1])
        filtered_data = data.loc[mask]
        st.dataframe(filtered_data[['Open', 'High', 'Low', 'Close', 'Volume']])
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Close'], mode='lines', name='Close'))
        fig_hist.update_layout(title="Historical Closing Prices", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write("Please select a valid date range.")

# Prediction History Tab
with tabs[3]:
    st.header("Prediction History")
    # For demonstration, let's assume you keep predictions in a DataFrame.
    # In a real app, you'd append new predictions to a persistent storage.
    if 'pred_history' not in st.session_state:
        st.session_state.pred_history = pd.DataFrame(columns=["Date", "Actual Close", "Predicted Close", "Signal"])
    
    # Save today's prediction
    today = pd.to_datetime(datetime.date.today())
    new_entry = pd.DataFrame({
        "Date": [today],
        "Actual Close": [current_close],
        "Predicted Close": [predicted_price],
        "Signal": [signal]
    })
    st.session_state.pred_history = pd.concat([st.session_state.pred_history, new_entry], ignore_index=True)
    
    st.dataframe(st.session_state.pred_history)



# In[ ]:




