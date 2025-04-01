#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta # Import pandas_ta
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Prediction Dashboard")

# --- Constants ---
TICKERS = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
MODEL_PATH = "models/" # Make sure this path is correct
FEATURES = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
TARGET = "Target" # Next day's closing price

# --- Model and Scaler Loading (Cached) ---
# Use st.cache_resource for objects that shouldn't be recreated often like models/scalers
@st.cache_resource
def load_model_scaler(ticker):
    """Loads the SVR model and scaler for the given ticker."""
    try:
        model = joblib.load(f"{MODEL_PATH}{ticker}_svr_model.pkl")
        scaler = joblib.load(f"{MODEL_PATH}{ticker}_scaler_X_svr.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model or scaler file not found for {ticker}. Please ensure files exist in '{MODEL_PATH}'.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/scaler for {ticker}: {e}")
        return None, None

# --- Data Fetching and Feature Engineering (Cached) ---
# Use st.cache_data for data that can be recalculated if the inputs change
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_data(ticker, period="1y"):
    """Fetches historical data from Yahoo Finance."""
    try:
        stock_data = yf.Ticker(ticker)
        hist_data = stock_data.history(period=period)
        # Ensure data is sorted by date (yf usually does this, but good practice)
        hist_data = hist_data.sort_index()
        # Convert index to DatetimeIndex if it's not already (handling potential issues)
        if not isinstance(hist_data.index, pd.DatetimeIndex):
             hist_data.index = pd.to_datetime(hist_data.index)
        return hist_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker} from yfinance: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

@st.cache_data
def calculate_features(df):
    """Calculates the required features for the model."""
    if df.empty:
        return df

    df_feat = df.copy()

    # 1. Lags
    df_feat['Lag1'] = df_feat['Close'].shift(1)
    df_feat['Lag7'] = df_feat['Close'].shift(7)

    # 2. Simple Moving Average (SMA) - using pandas_ta
    df_feat.ta.sma(length=7, append=True) # Appends 'SMA_7' column
    df_feat.rename(columns={'SMA_7': 'SMA7'}, inplace=True) # Rename if needed

    # 3. Relative Strength Index (RSI) - using pandas_ta
    df_feat.ta.rsi(length=14, append=True) # Appends 'RSI_14' column
    df_feat.rename(columns={'RSI_14': 'RSI14'}, inplace=True) # Rename if needed

    # 4. Moving Average Convergence Divergence (MACD) - using pandas_ta
    # This typically adds MACD_12_26_9, MACDh_12_26_9 (Histogram), MACDs_12_26_9 (Signal)
    df_feat.ta.macd(append=True)
    # Select the main MACD line (adjust column name based on pandas_ta version if needed)
    if 'MACD_12_26_9' in df_feat.columns:
         df_feat.rename(columns={'MACD_12_26_9': 'MACD'}, inplace=True)
    else:
         st.warning("Could not find 'MACD_12_26_9'. Check pandas_ta output.")
         df_feat['MACD'] = 0 # Assign default if not found

    # 5. Returns
    df_feat['Returns'] = df_feat['Close'].pct_change()
    # Or use pandas_ta: df_feat.ta.percent_return(append=True)

    # 6. Volatility (e.g., rolling standard deviation of returns)
    df_feat['Volatility'] = df_feat['Returns'].rolling(window=7).std() # 7-day volatility

    # Handle potential NaNs created by shifts and rolling calculations
    df_feat.dropna(inplace=True)

    return df_feat

# --- Prediction Function ---
def make_prediction(model, scaler, features_df, feature_list):
    """Makes prediction using the loaded model, scaler, and latest features."""
    if model is None or scaler is None or features_df.empty:
        return None

    # Ensure we have the required features in the correct order
    latest_features = features_df.iloc[[-1]][feature_list] # Get the last row

    # Check if latest_features still contains NaNs after processing
    if latest_features.isnull().values.any():
        st.warning("NaN values detected in the features for the latest data point. Prediction might be unreliable.")
        # Option 1: Fill NaNs (e.g., with 0 or forward fill) - Requires careful consideration
        # latest_features = latest_features.fillna(0)
        # Option 2: Skip prediction
        return None

    # Scale the features
    try:
        scaled_features = scaler.transform(latest_features)
    except Exception as e:
        st.error(f"Error scaling features: {e}")
        st.error(f"Features being scaled: \n{latest_features}") # Debug output
        return None

    # Make prediction
    try:
        prediction = model.predict(scaled_features)
        return prediction[0] # Return the single predicted value
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# --- Simple Buy/Sell Signal Logic ---
def get_signal(prediction, current_price):
    """Generates a simple signal based on prediction vs current price."""
    if prediction is None or current_price is None:
        return "Signal Unavailable", "⚪"

    signal_text = "Neutral / Hold"
    signal_color = "⚪" # White circle

    try:
        # Simple logic: Predict higher -> potential buy, predict lower -> potential sell
        price_diff = prediction - current_price
        percentage_diff = (price_diff / current_price) * 100 if current_price != 0 else 0

        # Add a small threshold to avoid noise (e.g., 0.5% change)
        if percentage_diff > 0.5:
            signal_text = f"Potential Buy Opportunity (Predicts +{percentage_diff:.2f}%)"
            signal_color = "🟢" # Green circle
        elif percentage_diff < -0.5:
             signal_text = f"Potential Sell/Caution (Predicts {percentage_diff:.2f}%)"
             signal_color = "🔴" # Red circle
        else:
            signal_text = f"Neutral / Hold (Predicted change {percentage_diff:.2f}%)"
            signal_color = "⚪" # White circle

    except TypeError:
         signal_text = "Error comparing prices"
         signal_color = "⚪" # White circle

    return signal_text, signal_color


# --- Streamlit App Layout ---

# --- Sidebar ---
st.sidebar.title("📈 Crypto Dashboard Settings")
selected_ticker = st.sidebar.selectbox("Select Crypto Ticker:", TICKERS)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**About:**
This dashboard predicts the next day's closing price for selected cryptocurrencies using pre-trained Support Vector Regression (SVR) models.

**Features Used:**
* Open, High, Low, Close, Volume
* Lag1, Lag7 (Previous day/week close)
* SMA7 (7-day Simple Moving Average)
* RSI14 (14-day Relative Strength Index)
* MACD (Moving Average Conv./Divergence)
* Returns (Daily Percentage Change)
* Volatility (7-day Rolling Std Dev)
""")
st.sidebar.markdown("---")
st.sidebar.info("⚠️ **Disclaimer:** Predictions and signals are for informational purposes only and NOT financial advice.")


# --- Main Area ---
st.title(f"{selected_ticker} Price Prediction Dashboard")

# Load Model and Scaler
model, scaler = load_model_scaler(selected_ticker)

# Fetch Data
data_hist = fetch_data(selected_ticker, period="1y") # Fetch 1 year for indicators

if not data_hist.empty:
    # Calculate Features
    data_features = calculate_features(data_hist)

    # --- Display Today's Market Data ---
    st.subheader("Current Market Snapshot")
    try:
        today_data = yf.Ticker(selected_ticker).history(period="2d") # Get last 2 days to calculate change
        if not today_data.empty:
            current_price = today_data['Close'].iloc[-1]
            prev_close = today_data['Close'].iloc[-2]
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100 if prev_close != 0 else 0
            volume_today = today_data['Volume'].iloc[-1]

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:,.4f}", f"{price_change:,.4f} ({percent_change:.2f}%)")
            col2.metric("Volume", f"{volume_today:,.0f}")
            # Add more metrics if desired (e.g., Day's High/Low)
            day_high = today_data['High'].iloc[-1]
            day_low = today_data['Low'].iloc[-1]
            col3.metric("Day's Range", f"${day_low:,.4f} - ${day_high:,.4f}")

        else:
            st.warning("Could not fetch current market data.")
            current_price = data_features['Close'].iloc[-1] # Fallback to last historical close
            st.metric("Last Close Price", f"${current_price:,.4f}")

    except Exception as e:
         st.error(f"Error fetching/displaying current data: {e}")
         current_price = data_features['Close'].iloc[-1] if not data_features.empty else None # Fallback

    st.markdown("---")


    # --- Prediction Section ---
    st.subheader("Next Day Price Prediction")
    prediction = None
    signal_text = "Signal Unavailable"
    signal_color = "⚪"

    if model and scaler and not data_features.empty:
        # Make sure FEATURES are present in data_features columns before prediction
        missing_features = [f for f in FEATURES if f not in data_features.columns]
        if not missing_features:
            prediction = make_prediction(model, scaler, data_features, FEATURES)
            if prediction is not None:
                st.metric("Predicted Close Price (Next Trading Day)", f"${prediction:,.4f}")

                # Get signal based on prediction vs CURRENT price
                signal_text, signal_color = get_signal(prediction, current_price)

            else:
                st.warning("Could not generate prediction.")
        else:
            st.error(f"Missing required features after calculation: {missing_features}. Cannot predict.")


    # --- Buy/Sell Signal ---
    st.subheader("Illustrative Signal")
    if prediction is not None and current_price is not None:
         st.markdown(f"### {signal_color} {signal_text}")
         st.caption("Signal based on comparing the predicted next day close to the *current* price. Threshold: +/- 0.5% change.")
    else:
         st.markdown("### ⚪ Signal Unavailable")
    st.info("⚠️ Reminder: This signal is based purely on the model's output vs current price and is NOT investment advice.")
    st.markdown("---")


    # --- Interactive Visualizations ---
    st.subheader("Historical Data & Indicators")

    # Create subplots: 1 for price/SMA, 1 for volume, 1 for RSI, 1 for MACD
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.5, 0.15, 0.15, 0.2]) # Adjust row heights as needed

    # Plot 1: Price and SMA
    fig.add_trace(go.Candlestick(x=data_features.index,
                                open=data_features['Open'], high=data_features['High'],
                                low=data_features['Low'], close=data_features['Close'],
                                name='Price'), row=1, col=1)
    if 'SMA7' in data_features.columns:
        fig.add_trace(go.Scatter(x=data_features.index, y=data_features['SMA7'],
                                 mode='lines', name='SMA 7', line=dict(color='orange')), row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False) # Hide range slider for top plot

    # Plot 2: Volume
    fig.add_trace(go.Bar(x=data_features.index, y=data_features['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

    # Plot 3: RSI
    if 'RSI14' in data_features.columns:
        fig.add_trace(go.Scatter(x=data_features.index, y=data_features['RSI14'],
                                 mode='lines', name='RSI 14', line=dict(color='purple')), row=3, col=1)
        # Add RSI Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Plot 4: MACD
    if 'MACD' in data_features.columns and 'MACDs_12_26_9' in data_features.columns and 'MACDh_12_26_9' in data_features.columns:
        fig.add_trace(go.Scatter(x=data_features.index, y=data_features['MACD'],
                                 mode='lines', name='MACD Line', line=dict(color='blue')), row=4, col=1)
        fig.add_trace(go.Scatter(x=data_features.index, y=data_features['MACDs_12_26_9'],
                                 mode='lines', name='Signal Line', line=dict(color='red')), row=4, col=1)
        # Use different colors for positive/negative histogram bars
        colors = ['green' if val >= 0 else 'red' for val in data_features['MACDh_12_26_9']]
        fig.add_trace(go.Bar(x=data_features.index, y=data_features['MACDh_12_26_9'], name='MACD Histogram', marker_color=colors), row=4, col=1)


    # Update layout
    fig.update_layout(
        height=800, # Adjust height as needed
        title=f'{selected_ticker} Historical Price, Volume & Technical Indicators',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Assign specific y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    st.plotly_chart(fig, use_container_width=True)


    # --- "Shine" Factor: Historical Predictions vs Actual (Optional but Recommended) ---
    st.subheader("Model Performance: Recent Predictions vs Actual")
    st.caption("Visual check of how the model performed on recent historical data.")

    try:
        # Select recent data (e.g., last 60 days from the features df)
        recent_data = data_features.tail(60).copy()

        if not recent_data.empty and model and scaler:
             # Prepare features (ensure they are in the correct order)
            recent_features_scaled = scaler.transform(recent_data[FEATURES])
            # Predict on the recent historical data
            historical_predictions = model.predict(recent_features_scaled)

            # Create a comparison DataFrame
            # Shift actual 'Close' to align with the prediction for the *next* day
            comparison_df = pd.DataFrame({
                'Actual Close': recent_data['Close'].shift(-1), # Shift actual close UP by one day
                'Predicted Close': historical_predictions
            }, index=recent_data.index) # Index remains the date the prediction was *made* on

            comparison_df.dropna(inplace=True) # Remove last row where actual is NaN

            if not comparison_df.empty:
                 # Plot comparison
                 fig_compare = go.Figure()
                 fig_compare.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Actual Close'],
                                                 mode='lines', name='Actual Close', line=dict(color='royalblue')))
                 fig_compare.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Predicted Close'],
                                                 mode='lines', name='Predicted Close', line=dict(color='darkorange', dash='dash')))
                 fig_compare.update_layout(title=f'Recent Actual vs Predicted Close Prices for {selected_ticker}',
                                         xaxis_title='Date', yaxis_title='Price ($)', height=400)
                 st.plotly_chart(fig_compare, use_container_width=True)
            else:
                 st.info("Not enough data for historical comparison plot.")

    except Exception as e:
         st.warning(f"Could not generate historical comparison plot: {e}")


else:
    st.error(f"Could not fetch or process data for {selected_ticker}. Please check the ticker symbol and your internet connection.")

# Add footer or more info
st.markdown("---")
st.markdown("Dashboard created using Streamlit, yfinance, Plotly, Scikit-learn & Pandas_TA.")


# In[ ]:




