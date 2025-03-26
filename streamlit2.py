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

def fetch_current_price(ticker):
    """
    Placeholder for fetching the current price of a cryptocurrency.
    Replace with your actual API integration (e.g., Binance, Coinbase, Alpha Vantage).
    """
    # In a real application, you would fetch live data here
    # For now, let's use a random value
    return np.random.uniform(100, 50000)

def preprocess_input_data(data, scaler):
    """Scales the input data using the loaded scaler."""
    df = pd.DataFrame([data])
    scaled_data = scaler.transform(df)
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

def create_price_history_chart(historical_data, predicted_price, current_price, symbol):
    """
    Placeholder for creating a price history chart with predictions.
    You'll need to integrate with a data source for historical data.
    """
    fig = go.Figure()
    # Add historical data trace (replace with your actual data)
    if historical_data is not None and not historical_data.empty:
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'],
                                 mode='lines', name='Historical Price'))

    # Add current price point
    fig.add_trace(go.Scatter(x=[datetime.datetime.now()], y=[current_price],
                             mode='markers', name='Current Price',
                             marker=dict(size=10, color='blue')))

    # Add predicted price point
    fig.add_trace(go.Scatter(x=[datetime.datetime.now() + datetime.timedelta(days=1)],
                             y=[predicted_price],
                             mode='markers', name='Predicted Price',
                             marker=dict(size=10, color='green')))

    fig.update_layout(title=f"{symbol} Price History and Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price")
    return fig

# --- Streamlit Application ---

def main():
    st.title("Crypto Price Prediction Dashboard")
    st.markdown("Predicting next-day closing prices using your trained XGBoost model.")

    # --- Sidebar for Settings ---
    st.sidebar.header("Settings")
    available_tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]
    selected_ticker = st.sidebar.selectbox("Select Cryptocurrency", available_tickers)
    prediction_threshold = st.sidebar.slider("Buy/Sell Threshold (%)", 0.5, 5.0, 2.0, step=0.1)

    # --- Load the Model and Scaler ---
    model, scaler = load_model_and_scaler(selected_ticker)

    if model is None or scaler is None:
        return

    # --- Input Features ---
    st.sidebar.header("Enter Current Data")
    st.sidebar.markdown("Provide the current values for the following features:")
    open_price = st.sidebar.number_input("Open", value=0.0)
    high_price = st.sidebar.number_input("High", value=0.0)
    low_price = st.sidebar.number_input("Low", value=0.0)
    close_price = st.sidebar.number_input("Close", value=0.0)
    volume = st.sidebar.number_input("Volume", value=0.0)
    lag1 = st.sidebar.number_input("Lag1 (Previous Day's Target)", value=0.0)
    lag7 = st.sidebar.number_input("Lag7 (7 Days Ago's Target)", value=0.0)
    sma7 = st.sidebar.number_input("SMA7 (7-day Simple Moving Average)", value=0.0)
    rsi14 = st.sidebar.number_input("RSI14 (14-day Relative Strength Index)", value=50.0, min_value=0.0, max_value=100.0)
    macd = st.sidebar.number_input("MACD (Moving Average Convergence Divergence)", value=0.0)
    returns = st.sidebar.number_input("Returns (e.g., Percentage Change)", value=0.0)
    volatility = st.sidebar.number_input("Volatility", value=0.0)

    input_data = {
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price,
        "Volume": volume,
        "Lag1": lag1,
        "Lag7": lag7,
        "SMA7": sma7,
        "RSI14": rsi14,
        "MACD": macd,
        "Returns": returns,
        "Volatility": volatility,
    }

    # --- Fetch Current Price ---
    current_price = fetch_current_price(selected_ticker)
    st.subheader(f"Current Price of {selected_ticker}")
    st.metric(selected_ticker, f"${current_price:.2f}")

    # --- Prediction ---
    if st.button("Predict Next Day Price"):
        # Preprocess the input data
        preprocessed_input = preprocess_input_data(input_data, scaler)

        # Make Prediction
        predicted_price = predict_price(model, preprocessed_input)

        if predicted_price is not None:
            st.subheader("Prediction")
            st.metric("Predicted Next Day Closing Price", f"${predicted_price:.2f}")

            # Generate Buy/Sell Signal
            signal = generate_signal(predicted_price, current_price, prediction_threshold)
            st.markdown(f"**Signal:** <span style='color: {'green' if signal == 'BUY' else 'red' if signal == 'SELL' else 'orange'};'>{signal}</span>", unsafe_allow_html=True)

            # --- Optional: Display Price History Chart ---
            if st.checkbox("Show Price History Chart"):
                st.subheader(f"{selected_ticker} Price History")
                # Replace this with your actual historical data fetching
                # For demonstration, let's create a dummy historical dataframe
                now = datetime.datetime.now()
                dates = pd.to_datetime([now - datetime.timedelta(days=5), now - datetime.timedelta(days=4), now - datetime.timedelta(days=3), now - datetime.timedelta(days=2), now - datetime.timedelta(days=1)])
                prices = np.array([current_price * 0.98, current_price * 1.01, current_price * 0.99, current_price * 1.02, current_price])
                historical_df = pd.DataFrame({'Close': prices}, index=dates)
                price_chart = create_price_history_chart(historical_df, predicted_price, current_price, selected_ticker)
                st.plotly_chart(price_chart)
        else:
            st.warning("Could not generate prediction.")

    # --- Additional Jawdropping Features (Suggestions) ---

    st.sidebar.header("More Features")
    if st.sidebar.checkbox("Show Feature Importance"):
        st.subheader("Feature Importance")
        # You would need to load the model again here to access feature_importances_
        temp_model, _ = load_model_and_scaler(selected_ticker)
        if temp_model:
            importance = temp_model.feature_importances_
            feature_names = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            st.dataframe(importance_df.sort_values(by='Importance', ascending=False))

    if st.sidebar.checkbox("About the Model"):
        st.subheader("About the Model")
        st.write("This application uses an XGBoost Regressor model trained on historical cryptocurrency data to predict the next day's closing price.")
        st.write("The model was trained using features like Open, High, Low, Close prices, Volume, Lagged target values, Simple Moving Average, Relative Strength Index, MACD, Returns, and Volatility.")

    if st.sidebar.checkbox("Disclaimer"):
        st.subheader("Disclaimer")
        st.warning("This is not financial advice. Cryptocurrency prices are highly volatile, and predictions are not guaranteed to be accurate. Use this tool for informational purposes only.")

if __name__ == "__main__":
    main()


# In[ ]:




