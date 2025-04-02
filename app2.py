#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime
import plotly.graph_objects as go
import ta

# Define available cryptocurrencies
tickers = {
    "BNB": "BNB-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XRP": "XRP-USD"
}

def load_model(ticker_short):
    model = joblib.load(f"{ticker}_svr_model.pkl")
    scaler = joblib.load(f"{ticker}_scaler_X_svr.pkl")
    return model, scaler

def fetch_crypto_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, interval="1d")
    df.reset_index(inplace=True)
    return df

def preprocess_data(df):
    df['Lag1'] = df['Close'].shift(1)
    df['Lag7'] = df['Close'].shift(7)
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=7).std()
    df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"]) 
    df["MACD"] = macd.macd()
    df.dropna(inplace=True)
    return df

def predict_next_close(model, scaler, df):
    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
    latest_data = df.iloc[-1][features].values.reshape(1, -1)
    latest_data_scaled = scaler.transform(latest_data)
    predicted_price = model.predict(latest_data_scaled)[0]
    return predicted_price

# Streamlit App UI
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("Crypto Price Prediction Dashboard")

crypto_choice = st.sidebar.selectbox("Select Cryptocurrency", list(tickers.keys()))
selected_ticker = tickers[crypto_choice]

st.sidebar.subheader("Select Historical Data Period")
period = st.sidebar.select_slider("Period", options=["1mo", "3mo", "6mo", "1y", "2y"], value="1y")

st.sidebar.subheader("Fetch Data")
if st.sidebar.button("Load Data"):
    data = fetch_crypto_data(selected_ticker, period)
    st.session_state["crypto_data"] = preprocess_data(data)

if "crypto_data" in st.session_state:
    df = st.session_state["crypto_data"]
    st.subheader(f"Historical Closing Prices for {crypto_choice}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price'))
    st.plotly_chart(fig, use_container_width=True)

    model, scaler = load_model(crypto_choice)
    predicted_price = predict_next_close(model, scaler, df)
    st.subheader(f"Predicted Next Day Close: ${predicted_price:.2f}")
    
    current_close = df['Close'].iloc[-1]
    signal = "Buy" if predicted_price > current_close else "Sell"
    st.subheader(f"Trading Signal: {signal}")

    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = pd.DataFrame(columns=["Date", "Actual Close", "Predicted Close", "Signal"])
    
    today = datetime.date.today()
    new_entry = pd.DataFrame({"Date": [today], "Actual Close": [current_close], "Predicted Close": [predicted_price], "Signal": [signal]})
    st.session_state["prediction_history"] = pd.concat([st.session_state["prediction_history"], new_entry], ignore_index=True)
    
    st.subheader("Prediction History")
    st.dataframe(st.session_state["prediction_history"])
else:
    st.warning("Click 'Load Data' to fetch and process historical crypto data.")


# In[ ]:




