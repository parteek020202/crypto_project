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
st.set_page_config(page_title="Crypto Forecast Pro", layout="wide", page_icon="🪙")

# Supported cryptocurrencies
CRYPTO_LIST = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Cache data loading
@st.cache_data(ttl=3600)
def load_data(ticker, days=60):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days + 30)  # Buffer for short timeframes
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            st.warning(f"No data returned for {ticker}")
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # Standardize column names
        column_mapping = {
            'open': 'Open', 'Open': 'Open',
            'high': 'High', 'High': 'High',
            'low': 'Low', 'Low': 'Low',
            'close': 'Close', 'Close': 'Close',
            'volume': 'Volume', 'Volume': 'Volume',
            'adj close': 'Adj Close', 'Adj Close': 'Adj Close'
        }
        df = df.rename(columns=column_mapping)
        
        # Verify required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns for {ticker}: {missing_columns}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None

# Technical indicators calculation
def calculate_features(df):
    if df is None or df.empty or len(df) < 14:
        return None
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing required columns in DataFrame: {required_columns}")
        return None
    
    df = df.copy()
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    try:
        df['RSI14'] = ta.momentum.RSIIndicator(df['Close'].squeeze(), window=14).rsi()
        macd = ta.trend.MACD(df['Close'].squeeze())
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=7).std()
        df['Lag1'] = df['Close'].shift(1)
        df['Lag7'] = df['Close'].shift(7)
        df['SMA7'] = df['Close'].rolling(window=7).mean()
        
        df = df.dropna()
        if len(df) < 7:
            st.warning(f"Insufficient data after processing: {len(df)} rows remaining")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error calculating features: {str(e)}")
        return None

# Prediction function
def make_prediction(ticker, data):
    if data is None or len(data) < 1:
        return None
        
    try:
        model = joblib.load(f"models/{ticker}_svr_model.pkl")
        scaler_X = joblib.load(f"scalers/{ticker}_scaler_X_svr.pkl")
    except FileNotFoundError:
        st.error(f"Model files not found for {ticker}")
        return None
    
    required_features = ["Open", "High", "Low", "Close", "Volume", 
                        "Lag1", "Lag7", "SMA7", "RSI14", "MACD", 
                        "Returns", "Volatility"]
    
    try:
        latest_data = data[required_features].iloc[[-1]]
        scaled_data = scaler_X.transform(latest_data)
        prediction = model.predict(scaled_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

#generate buy/sell/hold signals
def generate_signals(df, prediction, current_price):
    if df is None or len(df) < 2 or current_price == 0:
        return []
    
    signals = []
    try:
        latest = df.iloc[-1]
        
        # Get indicator values with NaN handling
        rsi = float(latest['RSI14']) if pd.notna(latest['RSI14']) else None
        close = float(latest['Close']) if pd.notna(latest['Close']) else None
        sma7 = float(latest['SMA7']) if pd.notna(latest['SMA7']) else None
        
        
        # Generate Buy, Sell, or Hold signal
        if rsi is not None and close is not None and sma7 is not None:
            # Buy: Prediction > current price, with RSI or SMA7 confirmation
            if (prediction is not None and prediction > current_price) and (rsi < 30 or close > sma7):
                signals.append(('Buy', 'success'))
            # Sell: Prediction < current price, with RSI or SMA7 confirmation
            elif (prediction is not None and prediction < current_price) and (rsi > 70 or close < sma7):
                signals.append(('Sell', 'danger'))
            # Hold: Neutral conditions or prediction close to current price
            else:
                signals.append(('Hold', 'info'))
            
    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
    
    return signals

# Create price chart
def create_price_chart(df, ticker):
    if df is None:
        return None
        
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05, 
                       row_heights=[0.6, 0.2, 0.2])
    
    # Price and SMA
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'],
                               high=df['High'], low=df['Low'],
                               close=df['Close'], name='Price'), 
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA7'], 
                           name='7D SMA', line=dict(color='blue')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], 
                           name='RSI 14', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=30, row=2, col=1, line_dash="dot", line_color="green")
    fig.add_hline(y=70, row=2, col=1, line_dash="dot", line_color="red")
    
    # MACD
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], 
                        name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                            name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], 
                            name='Signal', line=dict(color='red')), row=3, col=1)
    
    fig.update_layout(height=800, title=f"{ticker} Technical Analysis",
                     xaxis_rangeslider_visible=False)
    return fig

# Home Page
def show_home():
    st.title("Crypto Forecast Pro 🪙")
    st.subheader("Real-time Cryptocurrency Analysis & Prediction 📈 ")
    
    # Cryptocurrency selection
    selected_ticker = st.sidebar.selectbox("Select Cryptocurrency", CRYPTO_LIST)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["14 Days", "30 Days", "60 Days", "90 Days"]
    )
    days = int(timeframe.split()[0])
    
    # Load and process data
    with st.spinner("Loading market data..."):
        df = load_data(selected_ticker, days)
        if df is None:
            st.warning(f"No data available for {selected_ticker}")
            return
            
        df = calculate_features(df)
        if df is None:
            st.warning("Insufficient data for technical analysis")
            return
    
          # Make prediction
    with st.spinner("Generating prediction..."):
        prediction = make_prediction(selected_ticker, df)
    
    # Get current price
    latest_close = float(df['Close'].iloc[-1]) if df is not None and not df.empty else 0
    
    # Generate signals
    signals = generate_signals(df, prediction, latest_close)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${latest_close:.2f}")
    
    with col2:
        if prediction is not None and df is not None and not df.empty:
            prediction_change = ((prediction - latest_close) / latest_close) * 100
            st.metric("Tomorrow's Prediction", f"${prediction:.2f}", 
                     f"{prediction_change:.2f}%")
        else:
            st.metric("Tomorrow's Prediction", "N/A")
    
    with col3:
        st.subheader("Trading Signals")
        if signals:
            for signal, color in signals:
                color_text = "green" if color == "success" else "red"
                st.markdown(f"<span style='color:{color_text};'>● {signal}</span>", 
                           unsafe_allow_html=True)
        else:
            st.write("No signals available")
    
    # Price chart
    st.subheader("Technical Analysis")
    fig = create_price_chart(df, selected_ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Market Data
    st.subheader("Latest Market Data")
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

# Dashboard Page
def show_dashboard():
    st.title("Dashboard")
    
    # Cryptocurrency selection
    selected_ticker = st.sidebar.selectbox("Select Cryptocurrency", CRYPTO_LIST)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        ["14 Days", "30 Days", "60 Days", "90 Days"]
    )
    days = int(timeframe.split()[0])
    
    # Load and process data
    with st.spinner("Loading dashboard data..."):
        df = load_data(selected_ticker, days)
        if df is None:
            st.warning(f"No data available for {selected_ticker}")
            return
            
        df = calculate_features(df)
        if df is None:
            st.warning("Insufficient data for analysis")
            return
    
    # Dashboard tabs
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Technical Indicators", "Price Distribution"])
    
    with tab1:
        st.subheader("Price Movement Analysis")
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        )])
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA7'],
            line=dict(color='blue', width=1),
            name='SMA 7'
        ))
        fig.update_layout(
            title=f"{selected_ticker} Price Analysis",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily returns
        st.subheader("Daily Returns")
        returns_fig = go.Figure(data=[go.Bar(
            x=df.index,
            y=df['Returns'] * 100,
            marker_color=['red' if r < 0 else 'green' for r in df['Returns']],
            name='Daily Returns (%)'
        )])
        returns_fig.update_layout(
            title="Daily Returns (%)",
            yaxis_title="Return (%)",
            xaxis_title="Date"
        )
        st.plotly_chart(returns_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Technical Indicators")
        
        # RSI Chart
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI14'],
            line=dict(color='purple', width=2),
            name='RSI 14'
        ))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        rsi_fig.update_layout(
            title="Relative Strength Index (RSI)",
            yaxis_title="RSI Value",
            xaxis_title="Date"
        )
        st.plotly_chart(rsi_fig, use_container_width=True)
        
        # MACD Chart
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker_color=['red' if val < 0 else 'green' for val in df['MACD_Hist']]
        ))
        macd_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            line=dict(color='blue', width=2),
            name='MACD Line'
        ))
        macd_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            line=dict(color='red', width=1),
            name='Signal Line'
        ))
        macd_fig.update_layout(
            title="Moving Average Convergence Divergence (MACD)",
            yaxis_title="Value",
            xaxis_title="Date"
        )
        st.plotly_chart(macd_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Price Distribution Analysis")
        
        # Price histogram
        hist_fig = go.Figure(data=[go.Histogram(
            x=df['Close'],
            nbinsx=30,
            marker_color='blue',
            opacity=0.7
        )])
        hist_fig.update_layout(
            title="Closing Price Distribution",
            xaxis_title="Price (USD)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

# Model Information Page
def show_model_info():
    st.title("Model Information")
    st.markdown("---")

    st.header("📌 Model Overview")
    st.markdown("""
    This cryptocurrency forecasting model uses **Support Vector Regression (SVR)**, a powerful algorithm for regression tasks.
    The model is trained individually for four cryptocurrencies: **BNB, BTC, ETH, and XRP**.
    
    Key characteristics:
    - **Feature Scaling**: StandardScaler
    - **Hyperparameter Tuning**: Optuna(30 trials per asset)
    - **Evaluation Metrics**: MSE, MAPE, and Directional Accuracy
    - **Target**: Future price of the asset
    """)

    st.header("⚙️ Feature Engineering")

    st.markdown("""
    To enhance the predictive power of the model, I combined **basic market features** with carefully selected **technical indicators** derived from   historical price data.

    **Basic Features:**
    - `Open`, `High`, `Low`, `Close`, `Volume`: Core OHLCV data representing raw trading activity.

    **Engineered Technical Indicators:**
    - `Lag1`, `Lag7`: Lagged closing prices (1 and 7 days) to capture temporal dependencies.
    - `SMA7`: 7-day Simple Moving Average to smooth short-term price fluctuations.
    - `RSI14`: 14-day Relative Strength Index, used to assess momentum and overbought/oversold signals.
    - `MACD`: Moving Average Convergence Divergence, a momentum indicator that shows trend direction and strength.
    - `Returns`: Daily return percentage, capturing day-to-day price changes.
    - `Volatility`: Rolling standard deviation of returns, representing market risk and uncertainty.
    """)


    st.header("🔍 Hyperparameters (Best Found via Optuna)")
    st.markdown("""
    | Ticker   | Kernel | C       | Epsilon | Gamma     |
    |----------|--------|----------|---------|-----------|
    | BNB-USD  | rbf    | 689.08   | 0.0452  | 0.00038   |
    | BTC-USD  | rbf    | 951.53   | 0.0179  | 0.00187   |
    | ETH-USD  | rbf    | 46.55    | 0.0357  | 0.00630   |
    | XRP-USD  | rbf    | 1.93     | 0.0124  | 0.00010   |
    """)

    st.header("📊 Evaluation Results")
    st.markdown("""
    | Ticker   | MSE         | MAPE (%) | Directional Accuracy (%) |
    |----------|-------------|----------|---------------------------|
    | BNB-USD  | 374.12      | 2.17     | 50.00                     |
    | BTC-USD  | 272,278,179 | 16.70    | 45.00                     |
    | ETH-USD  | 20,739.29   | 3.61     | 51.43                     |
    | XRP-USD  | 0.84        | 29.55    | 52.14                     |
    """)

    st.info("""
    **Interpretation Notes:**
    - **MAPE** indicates prediction error as a percentage of actual values.
    - **Directional Accuracy** measures how well the model predicts price movement direction (up/down).
    - Lower MAPE and higher directional accuracy are preferred.
    """)


# Sidebar navigation
def sidebar_navigation():
    st.sidebar.title("Crypto Forecast Pro 🪙")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Dashboard", "Model Information"]
    )
    
    st.session_state['page'] = {
        "Home": "Home",
        "Dashboard": "Dashboard",
        "Model Information": "Model Info"
    }[page]
    
    st.sidebar.markdown("---")
    st.sidebar.caption("By Parteek Sharma")

# Main app
def main():
    sidebar_navigation()
    
    if st.session_state['page'] == 'Home':
        show_home()
    elif st.session_state['page'] == 'Dashboard':
        show_dashboard()
    elif st.session_state['page'] == 'Model Info':
        show_model_info()

if __name__ == "__main__":
    main()