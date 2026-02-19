# pro_stock_ai.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import ta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🇱🇰 PRO Sri Lankan AI Trading System")

# -------------------------
# USER INPUT
# -------------------------
symbol = st.text_input("Enter CSE Stock (ex: NABIL.N0000)", "NABIL.N0000")
start = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end   = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Run PRO Analysis"):

    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        st.error("No Data Found. Check Symbol.")
        st.stop()

    # -------------------------
    # Technical Indicators
    # -------------------------
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    # -------------------------
    # AI Prediction
    # -------------------------
    df["Day"] = np.arange(len(df))
    X = df[["Day"]]
    y = df["Close"]

    model = RandomForestRegressor(n_estimators=300)
    model.fit(X, y)

    future_days = np.arange(len(df), len(df)+30).reshape(-1,1)
    predictions = model.predict(future_days)

    future_dates = pd.date_range(df.index[-1], periods=31, freq="B")[1:]
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predictions
    }).set_index("Date")

    # -------------------------
    # Signal Engine
    # -------------------------
    latest_rsi = df["RSI"].iloc[-1]
    latest_price = df["Close"].iloc[-1]
    latest_ema = df["EMA_20"].iloc[-1]

    signal = "HOLD"

    if latest_rsi < 30 and latest_price > latest_ema:
        signal = "BUY ✅"
    elif latest_rsi > 70:
        signal = "SELL 🔻"

    # -------------------------
    # Risk Management
    # -------------------------
    stop_loss = latest_price * 0.95
    take_profit = latest_price * 1.10

    # -------------------------
    # DASHBOARD DISPLAY
    # -------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"LKR {latest_price:.2f}")
    col2.metric("RSI (14)", f"{latest_rsi:.2f}")
    col3.metric("Signal", signal)

    st.subheader("Technical Chart")

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["SMA_50"], label="SMA 50")
    ax.plot(df.index, df["EMA_20"], label="EMA 20")
    ax.legend()
    st.pyplot(fig)

    st.subheader("AI Predicted Next 30 Days")
    st.line_chart(future_df["Predicted_Close"])

    st.subheader("Risk Management")
    st.write(f"📉 Stop Loss: LKR {stop_loss:.2f}")
    st.write(f"🎯 Take Profit: LKR {take_profit:.2f}")

    st.download_button(
        "Download Prediction CSV",
        future_df.to_csv().encode("utf-8"),
        file_name=f"{symbol}_prediction.csv"
    )

    st.success("Professional Analysis Completed 🚀")
