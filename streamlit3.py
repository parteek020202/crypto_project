#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # For technical analysis indicators (SMA, RSI, MACD)

# --- Helper Functions ---

def load_model_and_scaler(ticker):
    """Loads the trained model and scaler for a given ticker."""
    try:
        model = joblib.load(f"{ticker}_xgboost_model.pkl")
        scaler = joblib.load(f"{ticker}_scaler_X_xgb.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found for {ticker}")
        return None, None

def load_processed_data(ticker):
    """Loads the processed data CSV for a given ticker."""
    try:
        df = pd.read_csv(f"{ticker}_processed.csv", index_col="Date", parse_dates=True)
        df = df.sort_index()
        return df
    except FileNotFoundError:
        st.error(f"Error: Processed data file not found for {ticker}")
        return None

def calculate_features(df):
    """Calculates the technical features needed by the model (optional, if already in CSV)."""
    if 'Lag1' not in df.columns:
        df['Lag1'] = df['Close'].shift(1).fillna(0)
    if 'Lag7' not in df.columns:
        df['Lag7'] = df['Close'].shift(7).fillna(0)
    if 'SMA7' not in df.columns:
        df['SMA7'] = ta.trend.sma_indicator(df['Close'], window=7).fillna(0)
    if 'RSI14' not in df.columns:
        df['RSI14'] = ta.momentum.rsi(df['Close'], window=14).fillna(50)
    if 'MACD' not in df.columns:
        macd_indicator = ta.trend.MACD(df['Close']).fillna(0)
        df['MACD'] = macd_indicator.macd()
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change().fillna(0) * 100
    if 'Volatility' not in df.columns:
        df['Volatility'] = df['Returns'].rolling(window=14).std().fillna(0)
    return df.dropna()

def preprocess_input_data(df, scaler, feature_names):
    """Preprocesses the input data using the loaded scaler."""
    df_processed = df[feature_names].tail(1) # Get the latest row
    scaled_data = scaler.transform(df_processed)
    return scaled_data

def predict_price(model, preprocessed_data):
    """Makes a prediction using the loaded model."""
    if model is not None:
        try:
            prediction = model.predict(preprocessed_data)
            return prediction[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    return None

def generate_signal(predicted_price, current_price, threshold_percent=2):
    """Generates a buy/sell signal based on the prediction."""
    if predicted_price > current_price * (1 + threshold_percent / 100):
        return "BUY"
    elif predicted_price < current_price * (1 - threshold_percent / 100):
        return "SELL"
    else:
        return "HOLD"

def create_interactive_price_chart(df, predicted_price, current_price, symbol):
    """Creates an interactive price chart using Plotly."""
    fig = go.Figure()

    # Historical Price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                             mode='lines', name='Historical Price',
                             line=dict(color='blue')))

    # Current Price (last available)
    fig.add_trace(go.Scatter(x=[df.index[-1]], y=[current_price],
                             mode='markers', name='Current Price (Last Available)',
                             marker=dict(size=10, color='orange')))

    # Predicted Price
    prediction_date = df.index[-1] + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(x=[prediction_date], y=[predicted_price],
                             mode='markers', name='Predicted Price',
                             marker=dict(size=10, color='green')))

    fig.update_layout(title=f"{symbol} Price History and Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=True)
    return fig

# --- Streamlit Application ---

def main():
    st.title("Advanced Crypto Price Prediction Dashboard")
    st.markdown("Predictions based on your processed historical data.")

    # --- Sidebar for Settings ---
    st.sidebar.header("Settings")
    available_tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
    selected_ticker = st.sidebar.selectbox("Select Cryptocurrency", available_tickers)
    prediction_threshold = st.sidebar.slider("Buy/Sell Threshold (%)", 0.5, 5.0, 2.0, step=0.1)

    # --- Load the Model and Scaler ---
    model, scaler = load_model_and_scaler(selected_ticker)
    if model is None or scaler is None:
        return

    # --- Load Processed Data ---
    processed_df = load_processed_data(selected_ticker)
    if processed_df is not None:
        # --- Calculate Features (Optional - if your CSVs already have them, you can skip this) ---
        processed_df = calculate_features(processed_df.copy())

        if not processed_df.empty:
            # --- Get Current Price (last available) ---
            current_price = processed_df['Close'].iloc[-1]
            st.subheader(f"Last Available Price of {selected_ticker}")
            st.metric(selected_ticker, f"${current_price:.2f}")

            # --- Prepare Data for Prediction ---
            feature_names = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
            if all(col in processed_df.columns for col in feature_names):
                preprocessed_data = preprocess_input_data(processed_df, scaler, feature_names)

                # --- Make Prediction ---
                predicted_price = predict_price(model, preprocessed_data)

                if predicted_price is not None:
                    st.subheader("Prediction")
                    st.metric("Predicted Next Day Closing Price", f"${predicted_price:.2f}")

                    # --- Generate Buy/Sell Signal ---
                    signal = generate_signal(predicted_price, current_price, prediction_threshold)
                    st.markdown(f"**Signal:** <span style='color: {'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'orange'};'>{signal}</span>", unsafe_allow_html=True)

                    # --- Interactive Price Chart ---
                    st.subheader(f"{selected_ticker} Price History and Prediction")
                    price_chart = create_interactive_price_chart(processed_df, predicted_price, current_price, selected_ticker)
                    st.plotly_chart(price_chart)

                else:
                    st.warning("Could not generate prediction.")
            else:
                st.error("Error: Not all required features found in the processed data.")
        else:
            st.warning("No data available in the processed file.")
    else:
        st.error("Failed to load processed data.")

    # --- Additional Features ---
    st.sidebar.header("More Features")
    if st.sidebar.checkbox("Show Feature Importance"):
        st.subheader("Feature Importance")
        temp_model, _ = load_model_and_scaler(selected_ticker)
        if temp_model:
            importance = temp_model.feature_importances_
            feature_names = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            st.dataframe(importance_df.sort_values(by='Importance', ascending=False))

    if st.sidebar.checkbox("About the Model"):
        st.subheader("About the Model")
        st.write("This application uses an XGBoost Regressor model trained on historical cryptocurrency data to predict the next day's closing price.")
        st.write("The model was trained using features like Open, High, Low prices, Volume, Lagged target values, Simple Moving Average, Relative Strength Index, MACD, Returns, and Volatility.")

    if st.sidebar.checkbox("Disclaimer"):
        st.subheader("Disclaimer")
        st.warning("This is not financial advice. Cryptocurrency prices are highly volatile, and predictions are not guaranteed to be accurate. Use this tool for informational purposes only.")

if __name__ == "__main__":
    main()
    


# In[ ]:




