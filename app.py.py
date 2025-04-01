#!/usr/bin/env python
# coding: utf-8

# In[14]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Crypto Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .buy-signal {
        color: green;
        font-weight: bold;
    }
    .sell-signal {
        color: red;
        font-weight: bold;
    }
    .hold-signal {
        color: #ff9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models and scalers
@st.cache_resource
def load_models_and_scalers():
    tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
    models = {}
    scalers = {}
    
    for ticker in tickers:
        ticker_key = ticker.replace("-", "")
        model_filename = f"{ticker_key}_svr_model.pkl"
        scaler_filename = f"{ticker_key}_scaler_X_svr.pkl"
        
        try:
            models[ticker] = pickle.load(open(model_filename, 'rb'))
            scalers[ticker] = pickle.load(open(scaler_filename, 'rb'))
        except FileNotFoundError:
            st.warning(f"Model or scaler file for {ticker} not found. Please make sure they exist in the current directory.")
    
    return models, scalers

# Function to get live data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_live_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        if len(data) == 0:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to preprocess data
def preprocess_data(data):
    if data is None or len(data) == 0:
        return None
    
    df = data.copy()
    
    # Calculate features
    df['Lag1'] = df['Close'].shift(1)
    df['Lag7'] = df['Close'].shift(7)
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # Calculate Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Drop NA values
    df = df.dropna()
    
    return df

# Function to make predictions
def make_prediction(model, scaler, data, ticker):
    if data is None or len(data) == 0:
        return None, None
    
    # Get the last row for prediction
    last_row = data.iloc[-1:].copy()
    
    # Create feature set
    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
    X = last_row[features].values
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Calculate percent change
    current_price = last_row['Close'].values[0]
    percent_change = ((prediction - current_price) / current_price) * 100
    
    return prediction, percent_change

# Function to generate buy/sell signals
def generate_signal(data, prediction, percent_change):
    if data is None or len(data) == 0:
        return "N/A"
    
    current_price = data.iloc[-1]['Close']
    
    # Get MACD signal
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_signal = macd.iloc[-1] > signal_line.iloc[-1]
    
    # Get RSI signal
    rsi = data['RSI14'].iloc[-1]
    rsi_signal = (rsi < 30)  # Oversold
    rsi_sell = (rsi > 70)  # Overbought
    
    # Get price prediction signal
    prediction_signal = percent_change > 2  # If predicted to go up by more than 2%
    prediction_sell = percent_change < -2  # If predicted to go down by more than 2%
    
    # Combine signals
    buy_signals = sum([macd_signal, rsi_signal, prediction_signal])
    sell_signals = sum([not macd_signal, rsi_sell, prediction_sell])
    
    if buy_signals >= 2:
        return "BUY"
    elif sell_signals >= 2:
        return "SELL"
    else:
        return "HOLD"

# Function to create candlestick chart with indicators
def create_candlestick_chart(data, ticker, prediction=None):
    if data is None or len(data) == 0:
        return None
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add SMA7
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA7'],
        name='SMA (7 days)',
        line=dict(color='purple', width=1)
    ))
    
    # Add prediction point if available
    if prediction is not None:
        last_date = data.index[-1]
        next_date = last_date + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[next_date],
            y=[prediction],
            mode='markers',
            marker=dict(
                size=10,
                color='green' if prediction > data['Close'].iloc[-1] else 'red',
                symbol='star'
            ),
            name='Prediction'
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Price Chart with Prediction',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Function to create MACD chart
def create_macd_chart(data):
    if data is None or len(data) == 0:
        return None
    
    # Calculate MACD components
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal_line
    
    fig = go.Figure()
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=macd,
        name='MACD',
        line=dict(color='blue', width=2)
    ))
    
    # Add Signal line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=signal_line,
        name='Signal Line',
        line=dict(color='red', width=1)
    ))
    
    # Add Histogram
    colors = ['green' if val >= 0 else 'red' for val in histogram]
    fig.add_trace(go.Bar(
        x=data.index,
        y=histogram,
        name='Histogram',
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title='MACD Indicator',
        xaxis_title='Date',
        yaxis_title='Value',
        height=300,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Function to create RSI chart
def create_rsi_chart(data):
    if data is None or len(data) == 0:
        return None
    
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI14'],
        name='RSI',
        line=dict(color='purple', width=2)
    ))
    
    # Add overbought/oversold lines
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[70, 70],
        name='Overbought (70)',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[data.index[0], data.index[-1]],
        y=[30, 30],
        name='Oversold (30)',
        line=dict(color='green', width=1, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='RSI Indicator',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=300,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Function to create volume chart
def create_volume_chart(data):
    if data is None or len(data) == 0:
        return None
    
    fig = go.Figure()
    
    # Add volume bars
    colors = ['green' if data['Close'].iloc[i] > data['Close'].iloc[i-1] 
              else 'red' for i in range(1, len(data))]
    colors.insert(0, 'gray')  # For the first day
    
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    # Update layout
    fig.update_layout(
        title='Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Function to create comparison chart for all cryptos
def create_comparison_chart(tickers):
    comparison_data = {}
    start_date = datetime.now() - timedelta(days=90)
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date)
            if len(data) > 0:
                # Normalize to percentage change from start
                first_price = data['Close'].iloc[0]
                normalized = (data['Close'] / first_price - 1) * 100
                # Store as Series with dates as index
                comparison_data[ticker] = normalized
        except Exception as e:
            st.error(f"Error fetching comparison data for {ticker}: {e}")
    
    if not comparison_data:
        return None
    
    # Create figure directly from the data
    fig = go.Figure()
    
    for ticker, values in comparison_data.items():
        fig.add_trace(go.Scatter(
            x=values.index,
            y=values.values,
            name=ticker
        ))
    
    fig.update_layout(
        title='90-Day Price Comparison (% Change)',
        xaxis_title='Date',
        yaxis_title='Percent Change (%)',
        height=400
    )
    
    return fig

# Main app
def main():
    # Display header
    st.markdown('<div class="main-header">Cryptocurrency Price Prediction Dashboard</div>', unsafe_allow_html=True)
    
    # Load models and scalers
    models, scalers = load_models_and_scalers()
    
    # Available tickers
    tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Dashboard", "Deep Dive Analysis", "Model Performance", "About"])
    
    if page == "Dashboard":
        st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
        
        # Market overview - comparison chart
        comparison_chart = create_comparison_chart(tickers)
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)
        else:
            st.warning("Could not create comparison chart. Please check your internet connection.")
        
        # Quick stats for all cryptos
        st.markdown('<div class="sub-header">Latest Market Stats</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, ticker in enumerate(tickers):
            col = [col1, col2, col3, col4][i]
            
            with col:
                try:
                    # Get latest data
                    latest_data = yf.download(ticker, period="2d")
                    if len(latest_data) > 0:
                        current_price = latest_data['Close'].iloc[-1]
                        prev_price = latest_data['Close'].iloc[-2] if len(latest_data) > 1 else current_price
                        price_change = ((current_price - prev_price) / prev_price) * 100
                        
                        color = "green" if price_change >= 0 else "red"
                        arrow = "↑" if price_change >= 0 else "↓"
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>{ticker.split('-')[0]}</h3>
                            <h2>${current_price:.2f}</h2>
                            <p style="color:{color};">{arrow} {abs(price_change):.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not fetch data for {ticker}: {e}")
        
        # Select ticker for detailed view
        st.markdown('<div class="sub-header">Detailed Analysis</div>', unsafe_allow_html=True)
        selected_ticker = st.selectbox("Select Cryptocurrency", tickers)
        
        if selected_ticker in models and selected_ticker in scalers:
            # Get live data
            data = get_live_data(selected_ticker)
            
            if data is not None and len(data) > 0:
                # Preprocess data
                processed_data = preprocess_data(data)
                
                if processed_data is not None:
                    # Make prediction
                    prediction, percent_change = make_prediction(
                        models[selected_ticker], 
                        scalers[selected_ticker], 
                        processed_data, 
                        selected_ticker
                    )
                    
                    # Generate signal
                    signal = generate_signal(processed_data, prediction, percent_change)
                    
                    # Show prediction and signal
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>Current Price</h3>
                            <h2>${processed_data['Close'].iloc[-1]:.2f}</h2>
                            <p>Last updated: {processed_data.index[-1].strftime('%Y-%m-%d %H:%M')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        signal_color = "green" if signal == "BUY" else "red" if signal == "SELL" else "#ff9800"
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>Prediction (Next Day)</h3>
                            <h2>${prediction:.2f}</h2>
                            <p style="color:{'green' if percent_change >= 0 else 'red'};">
                                Expected change: {percent_change:.2f}%
                            </p>
                            <p>Signal: <span style="color:{signal_color}; font-weight:bold;">{signal}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show candlestick chart
                    candlestick_fig = create_candlestick_chart(processed_data, selected_ticker, prediction)
                    if candlestick_fig:
                        st.plotly_chart(candlestick_fig, use_container_width=True)
                    
                    # Indicators
                    st.markdown('<div class="sub-header">Technical Indicators</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # MACD Chart
                        macd_fig = create_macd_chart(processed_data)
                        if macd_fig:
                            st.plotly_chart(macd_fig, use_container_width=True)
                    
                    with col2:
                        # RSI Chart
                        rsi_fig = create_rsi_chart(processed_data)
                        if rsi_fig:
                            st.plotly_chart(rsi_fig, use_container_width=True)
                    
                    # Volume Chart
                    volume_fig = create_volume_chart(processed_data)
                    if volume_fig:
                        st.plotly_chart(volume_fig, use_container_width=True)
                else:
                    st.error("Could not process data for prediction.")
            else:
                st.error(f"Could not fetch data for {selected_ticker}. Please check your internet connection.")
        else:
            st.error(f"Model or scaler not found for {selected_ticker}.")
    
    elif page == "Deep Dive Analysis":
        st.markdown('<div class="sub-header">Deep Dive Analysis</div>', unsafe_allow_html=True)
        
        selected_ticker = st.selectbox("Select Cryptocurrency", tickers)
        time_period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        # Get and process data
        data = get_live_data(selected_ticker, period=time_period)
        
        if data is not None and len(data) > 0:
            processed_data = preprocess_data(data)
            
            if processed_data is not None:
                # Show tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Price Analysis", "Volume Analysis", "Correlation Analysis"])
                
                with tab1:
                    st.markdown("### Price Trend Analysis")
                    
                    # Price chart with moving averages
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=processed_data.index,
                        open=processed_data['Open'],
                        high=processed_data['High'],
                        low=processed_data['Low'],
                        close=processed_data['Close'],
                        name='Price'
                    ))
                    
                    # Add SMA7
                    fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=processed_data['SMA7'],
                        name='SMA (7 days)',
                        line=dict(color='purple', width=1)
                    ))
                    
                    # Add SMA30
                    sma30 = processed_data['Close'].rolling(window=30).mean()
                    fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=sma30,
                        name='SMA (30 days)',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Add SMA90
                    sma90 = processed_data['Close'].rolling(window=90).mean()
                    fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=sma90,
                        name='SMA (90 days)',
                        line=dict(color='green', width=1)
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_ticker} Price with Moving Averages',
                        height=600,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show volatility
                    st.markdown("### Volatility Analysis")
                    
                    vol_fig = go.Figure()
                    
                    vol_fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=processed_data['Volatility'] * 100,  # Convert to percentage
                        name='Volatility (20-day)',
                        line=dict(color='red', width=2)
                    ))
                    
                    vol_fig.update_layout(
                        title=f'{selected_ticker} Volatility (20-day rolling)',
                        xaxis_title='Date',
                        yaxis_title='Volatility (%)',
                        height=400
                    )
                    
                    st.plotly_chart(vol_fig, use_container_width=True)
                
                with tab2:
                    st.markdown("### Volume Analysis")
                    
                    # Volume chart with moving average
                    vol_fig = go.Figure()
                    
                    # Add volume bars
                    colors = ['green' if processed_data['Close'].iloc[i] > processed_data['Close'].iloc[i-1] 
                              else 'red' for i in range(1, len(processed_data))]
                    colors.insert(0, 'gray')  # For the first day
                    
                    vol_fig.add_trace(go.Bar(
                        x=processed_data.index,
                        y=processed_data['Volume'],
                        name='Volume',
                        marker_color=colors
                    ))
                    
                    # Add volume moving average
                    vol_ma = processed_data['Volume'].rolling(window=20).mean()
                    vol_fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=vol_ma,
                        name='Volume MA (20 days)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    vol_fig.update_layout(
                        title=f'{selected_ticker} Trading Volume',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        height=500
                    )
                    
                    st.plotly_chart(vol_fig, use_container_width=True)
                    
                    # Show volume to price ratio
                    vol_price_ratio = processed_data['Volume'] / processed_data['Close']
                    
                    vprice_fig = go.Figure()
                    
                    vprice_fig.add_trace(go.Scatter(
                        x=processed_data.index,
                        y=vol_price_ratio,
                        name='Volume/Price Ratio',
                        line=dict(color='purple', width=2)
                    ))
                    
                    vprice_fig.update_layout(
                        title=f'{selected_ticker} Volume to Price Ratio',
                        xaxis_title='Date',
                        yaxis_title='Volume/Price Ratio',
                        height=400
                    )
                    
                    st.plotly_chart(vprice_fig, use_container_width=True)
                
                with tab3:
                    st.markdown("### Correlation Analysis")
                    
                    st.info("This section shows how this cryptocurrency correlates with other major cryptocurrencies.")
                    
                    # Get data for all tickers for correlation analysis
                    corr_data = {}
                    
                    for ticker in tickers:
                        try:
                            data = get_live_data(ticker, period=time_period)
                            if data is not None and len(data) > 0:
                                corr_data[ticker.split('-')[0]] = data['Close']
                        except Exception as e:
                            st.error(f"Error fetching correlation data for {ticker}: {e}")
                    
                    if corr_data:
                        # Create correlation DataFrame
                        corr_df = pd.DataFrame(corr_data)
                        
                        # Calculate correlation matrix
                        corr_matrix = corr_df.corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            color_continuous_scale='Blues',
                            title='Correlation Matrix between Cryptocurrencies'
                        )
                        
                        fig.update_layout(height=500)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show rolling correlation with selected crypto
                        if len(tickers) > 1:
                            st.markdown("### Rolling Correlation")
                            
                            compare_options = [t.split('-')[0] for t in tickers if t != selected_ticker]
                            compare_with = st.selectbox("Compare with", compare_options)
                            window_size = st.slider("Window size (days)", 10, 90, 30)
                            
                            selected_symbol = selected_ticker.split('-')[0]
                            
                            if selected_symbol in corr_df.columns and compare_with in corr_df.columns:
                                # Calculate rolling correlation
                                rolling_corr = corr_df[selected_symbol].rolling(window=window_size).corr(corr_df[compare_with])
                                
                                # Create figure
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=rolling_corr.index,
                                    y=rolling_corr,
                                    name=f'Correlation: {selected_symbol} vs {compare_with}',
                                    line=dict(color='green', width=2)
                                ))
                                
                                fig.update_layout(
                                    title=f'{window_size}-day Rolling Correlation between {selected_symbol} and {compare_with}',
                                    xaxis_title='Date',
                                    yaxis_title='Correlation Coefficient',
                                    height=400,
                                    yaxis=dict(range=[-1, 1])
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not calculate correlation for selected cryptocurrencies.")
                    else:
                        st.warning("Could not fetch data for correlation analysis.")
            else:
                st.error("Could not process data for analysis.")
        else:
            st.error(f"Could not fetch data for {selected_ticker}. Please check your internet connection.")
    
    elif page == "Model Performance":
        st.markdown('<div class="sub-header">Model Performance Evaluation</div>', unsafe_allow_html=True)
        
        selected_ticker = st.selectbox("Select Cryptocurrency", tickers)
        
        st.info("""
        This section evaluates how well the SVR model predicts prices by comparing its predictions 
        against actual historical data. It shows the model's accuracy, error metrics, and visualizes 
        predicted vs. actual prices.
        """)
        
        if selected_ticker in models and selected_ticker in scalers:
            # Get data for backtesting
            data = get_live_data(selected_ticker, period="1y")
            
            if data is not None and len(data) > 0:
                # Preprocess data
                processed_data = preprocess_data(data)
                
                if processed_data is not None:
                    st.markdown("### Backtesting Model Predictions")
                    
                    # Create feature set for backtesting
                    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
                    X = processed_data[features].values
                    y_actual = processed_data['Close'].shift(-1).dropna().values  # Next day's close prices
                    
                    # Scale features
                    X_scaled = scalers[selected_ticker].transform(X[:-1])  # Remove last row since we don't have y for it
                    
                    # Make predictions
                    y_pred = models[selected_ticker].predict(X_scaled)
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                    
                    mae = mean_absolute_error(y_actual, y_pred)
                    mse = mean_squared_error(y_actual, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_actual, y_pred)
                    
                    # Create metrics columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>MAE</h3>
                            <h2>${mae:.2f}</h2>
                            <p>Mean Absolute Error</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>RMSE</h3>
                            <h2>${rmse:.2f}</h2>
                            <p>Root Mean Squared Error</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>MSE</h3>
                            <h2>${mse:.2f}</h2>
                            <p>Mean Squared Error</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        accuracy = r2 * 100
                        color = "green" if accuracy > 60 else "orange" if accuracy > 30 else "red"
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h3>R² Score</h3>
                            <h2 style="color:{color};">{accuracy:.2f}%</h2>
                            <p>Coefficient of Determination</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create prediction vs actual chart
                    fig = go.Figure()
                    
                    # Add actual prices
                    fig.add_trace(go.Scatter(
                        x=processed_data.index[:-1],
                        y=y_actual,
                        name='Actual Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add predicted prices
                    fig.add_trace(go.Scatter(
                        x=processed_data.index[:-1],
                        y=y_pred,
                        name='Predicted Price',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_ticker} - Actual vs. Predicted Prices',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals plot
                    residuals = y_actual - y_pred
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=processed_data.index[:-1],
                        y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(
                            color=residuals,
                            colorscale='RdBu',
                            colorbar=dict(title='Error (USD)'),
                            size=8
                        )
                    ))
                    
                    # Add zero line
                    fig.add_trace(go.Scatter(
                        x=[processed_data.index[0], processed_data.index[-2]],
                        y=[0, 0],
                        mode='lines',
                        line=dict(color='black', width=1, dash='dash'),
                        name='Zero Error'
                    ))
                    
                    fig.update_layout(
                        title='Prediction Errors (Residuals) Over Time',
                        xaxis_title='Date',
                        yaxis_title='Error (USD)',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Error distribution
                    fig = px.histogram(
                        residuals,
                        title='Distribution of Prediction Errors',
                        labels={'value': 'Prediction Error (USD)'},
                        color_discrete_sequence=['indianred']
                    )
                    
                    fig.add_vline(
                        x=0, 
                        line_dash="dash", 
                        line_color="black",
                        annotation_text="Perfect Prediction", 
                        annotation_position="top"
                    )
                    
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not process data for model evaluation.")
            else:
                st.error(f"Could not fetch data for {selected_ticker}. Please check your internet connection.")
        else:
            st.error(f"Model or scaler not found for {selected_ticker}.")
    
    elif page == "About":
        st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Crypto Price Prediction Dashboard
        
        This application uses Support Vector Regression (SVR) models to predict cryptocurrency prices based on historical data 
        and technical indicators. The models were trained on historical data for four major cryptocurrencies:
        
        - Bitcoin (BTC)
        - Ethereum (ETH)
        - Binance Coin (BNB)
        - XRP (Ripple)
        
        ### Features Used in Prediction
        
        The following features are used by the SVR models to make predictions:
        
        - **Price data**: Open, High, Low, Close
        - **Volume**: Trading volume
        - **Lag values**: Previous day's close (Lag1) and 7-day previous close (Lag7)
        - **Moving Averages**: 7-day Simple Moving Average (SMA7)
        - **Technical Indicators**: 14-day Relative Strength Index (RSI14), Moving Average Convergence Divergence (MACD)
        - **Price dynamics**: Daily returns and 20-day volatility
        
        ### How to Use This Dashboard
        
        - **Dashboard**: View current prices, predictions, and signals for all cryptocurrencies.
        - **Deep Dive Analysis**: Analyze price trends, volume patterns, and correlations between cryptocurrencies.
        - **Model Performance**: Evaluate the accuracy of the SVR models through backtesting.
        - **About**: Learn more about the project and the methods used.
        
        ### Trading Signals
        
        The dashboard generates trading signals (BUY, SELL, HOLD) based on a combination of:
        
        - Price prediction (next day)
        - MACD indicator
        - RSI indicator
        
        ### Technologies Used
        
        - **Data**: yfinance for live cryptocurrency data
        - **Models**: Support Vector Regression (SVR)
        - **Visualization**: Plotly and Matplotlib
        - **Frontend**: Streamlit
        
        ### Disclaimer
        
        This application is for educational purposes only. Cryptocurrency trading involves significant risks, and the predictions 
        and signals should not be considered as financial advice. Always do your own research before making investment decisions.
        """)

if __name__ == "__main__":
    main()
                


# In[ ]:




