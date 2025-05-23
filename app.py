import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
import requests
import datetime

st.set_page_config(page_title="📈 LSTM Stock Predictor", layout="centered")

@st.cache_resource
def load_model_once():
    return load_model("lstm_stock_model.h5")

@st.cache_data
def search_tickers(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        results = response.json().get("quotes", [])
        return [
            f"{item['symbol']} - {item.get('shortname') or item.get('longname') or item['symbol']}"
            for item in results
            if item.get("quoteType") == "EQUITY"
        ]
    return []

def add_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df['Close'].rolling(20).mean()
    df.fillna(method='bfill', inplace=True)
    return df

def prepare_data(df):
    data = df[['Close', 'RSI', 'MA20']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def predict_prices(model, X, scaler, y):
    preds = model.predict(X)
    padding = np.zeros((len(preds), 2))
    pred_full = np.concatenate((preds, padding), axis=1)
    actual_full = np.concatenate((y.reshape(-1, 1), padding), axis=1)
    predicted = scaler.inverse_transform(pred_full)[:, 0]
    actual = scaler.inverse_transform(actual_full)[:, 0]
    return predicted, actual

def forecast_next_days(model, df, scaler, days=7):
    recent_data = df[['Close', 'RSI', 'MA20']].values[-60:]
    scaled = scaler.transform(recent_data)
    forecast = []

    for _ in range(days):
        input_seq = np.array(scaled[-60:]).reshape(1, 60, 3)
        pred = model.predict(input_seq)[0][0]
        last_rsi_ma20 = scaled[-1, 1:]
        next_scaled = np.array([pred, last_rsi_ma20[0], last_rsi_ma20[1]])
        scaled = np.vstack([scaled, next_scaled])
        forecast.append(pred)

    forecast = np.array(forecast).reshape(-1, 1)
    padding = np.zeros((forecast.shape[0], 2))
    forecast_full = np.concatenate((forecast, padding), axis=1)
    return scaler.inverse_transform(forecast_full)[:, 0]

st.markdown("<h2 style='text-align:center;'>📊 LSTM Stock Price Predictor</h2>", unsafe_allow_html=True)

query = st.text_input("🔍 Search stock ticker or company name:", value="AAPL")

suggestions = search_tickers(query) if query else []

selected_full = st.selectbox("Select from results:", suggestions) if suggestions else None
selected_ticker = selected_full.split(" - ")[0] if selected_full else None

forecast_days = st.slider("📆 Forecast future days", 3, 30, 7)

if selected_ticker and st.button("🔮 Predict Now"):
    with st.spinner(f"Loading data for {selected_ticker}..."):
        df = yf.download(selected_ticker, start="2015-01-01", end=datetime.datetime.today().strftime('%Y-%m-%d'))
        df = add_indicators(df)
        X, y, scaler = prepare_data(df)
        model = load_model_once()
        predicted, actual = predict_prices(model, X, scaler, y)
        forecasted = forecast_next_days(model, df, scaler, forecast_days)

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        st.subheader(f"📈 Actual vs Predicted for {selected_ticker}")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(actual, label="Actual", color="blue")
        ax1.plot(predicted, label="Predicted", color="orange")
        ax1.set_title("Model Fit on Historical Data")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        st.subheader("🔢 Forecasted Prices")
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Price": forecasted.round(2)
        })
        st.dataframe(forecast_df)

        st.subheader(f"🔮 Forecast for Next {forecast_days} Days")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(future_dates, forecasted, marker='o', linestyle='--', color='green', label="Forecast")
        ax2.set_title("Future Price Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        st.success(f"✅ RMSE: {rmse:.2f}")
        st.info(f"📉 MAPE: {mape:.2f}%")

        st.subheader("📌 Recent RSI & MA20")
        st.dataframe(df[['Close', 'RSI', 'MA20']].tail(5).round(2))

st.markdown("---")
st.caption("⚙️ Built using LSTM · Keras · Streamlit · Yahoo Finance by APR")
