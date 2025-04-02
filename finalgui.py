#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# App configuration
st.set_page_config(page_title="Crypto Forecast", layout="wide", page_icon="🚀")

# Supported cryptocurrencies
CRYPTO_LIST = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]

# Cache data loading
@st.cache_data
def load_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Technical indicators calculation
def calculate_features(df):
    if df.empty:
        return df
    
    # Use pandas Series directly
    close_prices = df['Close']
    high_prices = df['High']
    low_prices = df['Low']
    volume = df['Volume']
    
    # Calculate indicators
    df['RSI14'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    macd = ta.trend.MACD(df['Close'].squeeze())
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Calculate other features
    df['Returns'] = close_prices.pct_change()
    df['Volatility'] = df['Returns'].rolling(window=7).std()
    df['Lag1'] = close_prices.shift(1)
    df['Lag7'] = close_prices.shift(7)
    df['SMA7'] = close_prices.rolling(window=7).mean()
    
    return df.dropna()

# Prediction function
def make_prediction(ticker, data):
    try:
        model = joblib.load(f"{ticker}_svr_model.pkl")
        scaler_X = joblib.load(f"{ticker}_scaler_X_svr.pkl")
    except FileNotFoundError:
        st.error(f"Model files not found for {ticker}")
        return None
    
    required_features = ["Open", "High", "Low", "Close", "Volume", 
                        "Lag1", "Lag7", "SMA7", "RSI14", "MACD", 
                        "Returns", "Volatility"]
    
    latest_data = data[required_features].iloc[[-1]]
    scaled_data = scaler_X.transform(latest_data)
    prediction = model.predict(scaled_data)
    return prediction[0]

# Buy/Sell signals
def generate_signals(df):
    if len(df) < 2:  # Ensure enough data points
        return []
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = []
    
    # RSI signal (convert to float explicitly)
    rsi = float(latest['RSI14'])
    if rsi < 30:
        signals.append(('RSI Buy', 'success'))
    elif rsi > 70:
        signals.append(('RSI Sell', 'danger'))
    
    # MACD signal
    macd = float(latest['MACD'])
    macd_signal = float(latest['MACD_Signal'])
    prev_macd = float(prev['MACD'])
    prev_signal = float(prev['MACD_Signal'])
    
    if macd > macd_signal and prev_macd < prev_signal:
        signals.append(('MACD Buy', 'success'))
    elif macd < macd_signal and prev_macd > prev_signal:
        signals.append(('MACD Sell', 'danger'))
    
    return signals
# Main app
def main():
    st.title("🚀 Crypto Forecasting Pro")
    st.markdown("### Real-time Cryptocurrency Analysis & Prediction")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_ticker = st.sidebar.selectbox("Select Cryptocurrency", CRYPTO_LIST)
    
    # Data loading and processing
    df = load_data(selected_ticker)
    df = calculate_features(df)
    
    if len(df) < 7:
        st.warning("Insufficient historical data for accurate predictions")
        return
    
   # Make prediction
    prediction = make_prediction(selected_ticker, df)
    if prediction is None:
        return

    # Extract latest_close as a scalar float
    latest_close = float(df['Close'].iloc[-1])  # Convert to float explicitly
    prediction_change = ((prediction - latest_close) / latest_close) * 100

    # Generate signals
    signals = generate_signals(df)

    # Dashboard layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${latest_close:.2f}")
    with col2:
        st.metric("Tomorrow's Prediction", f"${prediction:.2f}", 
                 f"{prediction_change:.2f}%")
    with col3:
        st.write("**Trading Signals**")
        for signal, color in signals:
            st.markdown(f"<span style='color:{color};'>◉ {signal}</span>", 
                       unsafe_allow_html=True)
        
    # Price and indicators visualization
    st.subheader("Technical Analysis")
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, 
                       row_heights=[0.6, 0.2, 0.2])
    
    # Price and SMA
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'],
                                high=df['High'], low=df['Low'],
                                close=df['Close'], name='Price'), 
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA7'], 
                           name='7D SMA'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], 
                           name='RSI 14'), row=2, col=1)
    fig.add_hline(y=30, row=2, col=1, line_dash="dot", 
                 line_color="green")
    fig.add_hline(y=70, row=2, col=1, line_dash="dot", 
                 line_color="red")
    
    # MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], 
                        name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                            name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], 
                            name='Signal'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False, 
                     xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Latest data table
    st.subheader("Latest Market Data")
    st.dataframe(df.tail(10).sort_index(ascending=False), 
                use_container_width=True)
    
    # Model information
    with st.expander("Model Details"):
        st.write("""
        **SVR Model Details:**
        - Trained on 2 years of historical data
        - Features used: OHLCV + Technical Indicators
        - RBF Kernel with optimized hyperparameters
        """)

if __name__ == "__main__":
    main()


# In[ ]:




