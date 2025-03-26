import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

tickers = ["BNB-USD", "BTC-USD", "ETH-USD", "XRP-USD"]

def load_model(ticker):
    model = joblib.load(f"{ticker}_xgboost_model.pkl")
    scaler = joblib.load(f"{ticker}_scaler_X_xgb.pkl")
    return model, scaler

def prepare_features(df):
    features = ["Open", "High", "Low", "Close", "Volume", "Lag1", "Lag7", "SMA7", "RSI14", "MACD", "Returns", "Volatility"]
    return df[features].values

def predict_prices(model, scaler, df):
    X_scaled = scaler.transform(prepare_features(df))
    predictions = model.predict(X_scaled)
    return predictions

def generate_signals(df):
    df["Signal"] = "Hold"
    df.loc[df["Predicted"].shift(-1) > df["Predicted"], "Signal"] = "Buy"
    df.loc[df["Predicted"].shift(-1) < df["Predicted"], "Signal"] = "Sell"
    return df

# Streamlit UI
st.title("Crypto Price Prediction with XGBoost")
st.sidebar.header("Select Cryptocurrency")
ticker = st.sidebar.selectbox("Choose a Crypto", tickers)

st.sidebar.header("Upload Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col="Date", parse_dates=True)
    st.write("### Uploaded Data Sample")
    st.write(df.head())
else:
    df = pd.read_csv(f"{ticker}_processed.csv", index_col="Date", parse_dates=True)
    st.write(f"### Using Default Dataset for {ticker}")
    st.write(df.tail())

# Load model and scaler
model, scaler = load_model(ticker)

# Predict Prices
df["Predicted"] = predict_prices(model, scaler, df)

# Generate Buy/Sell Signals
df = generate_signals(df)

# Display Metrics
st.write("### Model Performance Metrics")
mse = np.mean((df["Predicted"] - df["Target"]) ** 2)
mape = np.mean(np.abs((df["Predicted"] - df["Target"]) / df["Target"])) * 100
st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"- **Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# Plot Actual vs Predicted
fig = px.line(df, x=df.index, y=["Target", "Predicted"], labels={"value": "Price", "index": "Date"}, title=f"Actual vs Predicted Prices for {ticker}")
st.plotly_chart(fig)

# Feature Importance
importance = model.feature_importances_
feature_df = pd.DataFrame({"Feature": prepare_features(df).columns, "Importance": importance})
feature_df = feature_df.sort_values(by="Importance", ascending=False)
st.write("### Feature Importance")
st.bar_chart(feature_df.set_index("Feature"))

# Display Buy/Sell Signals
st.write("### Buy/Sell Signals")
st.write(df[["Predicted", "Signal"]].tail(10))
