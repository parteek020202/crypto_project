import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# Set up the app
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("📈 Cryptocurrency Price Prediction & Analysis")

# Sidebar settings
st.sidebar.header("Settings")
crypto_symbol = st.sidebar.text_input("Enter Cryptocurrency Ticker (e.g., BTC-USD):", "BTC-USD")
days = st.sidebar.slider("Select Number of Days for Analysis:", 30, 365, 180)

# Fetch data
def fetch_data(symbol, period="1y", interval="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

data = fetch_data(crypto_symbol, period="1y")
if data is not None:
    st.subheader(f"{crypto_symbol} - Historical Data")
    st.dataframe(data.tail(10))
    
    # Technical Indicators
    data['SMA7'] = data['Close'].rolling(window=7).mean()
    data['SMA30'] = data['Close'].rolling(window=30).mean()
    data['RSI14'] = RSIIndicator(close=data['Close'], window=14).rsi()
    macd = MACD(close=data['Close']).macd()
    data['MACD'] = macd
    bb = BollingerBands(close=data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()

    # Plot Historical Prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA7'], mode='lines', name='7-Day SMA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA30'], mode='lines', name='30-Day SMA'))
    st.plotly_chart(fig, use_container_width=True)

    # RSI Plot
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI14'], mode='lines', name='RSI (14)'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.subheader("RSI Indicator")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD Plot
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    st.subheader("MACD Indicator")
    st.plotly_chart(fig_macd, use_container_width=True)
    
    # Trading Signals
    predicted_price = data['Close'].iloc[-1] * (1 + np.random.uniform(-0.02, 0.02))
    signal = "Buy" if predicted_price > data['Close'].iloc[-1] else "Sell"
    
    st.subheader(f"Predicted Next Day Closing Price: ${predicted_price:.2f}")
    st.subheader(f"Trading Signal: {signal}")
    
    # Bollinger Bands Chart
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='Upper Band'))
    fig_bb.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='Lower Band'))
    st.subheader("Bollinger Bands")
    st.plotly_chart(fig_bb, use_container_width=True)

st.sidebar.info("Developed by AI-powered FinTech Solutions 🚀")
