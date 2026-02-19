import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("🇱🇰 CSE AI PRO ULTIMATE")

# --------------------------------
# SETTINGS
# --------------------------------
STOCK_LIST = ["JKH.CM", "SAMP.CM", "COMB.CM", "HAYL.CM", "LOLC.CM"]

# --------------------------------
# GET YAHOO DATA
# --------------------------------
def get_data(symbol):
    try:
        data = yf.download(symbol, period="6mo", interval="1d")
        return data
    except:
        return None

# --------------------------------
# RSI
# --------------------------------
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --------------------------------
# AI PREDICTION
# --------------------------------
def predict_price(data):
    data = data.dropna()
    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_day = [[len(data) + 5]]
    prediction = model.predict(future_day)
    return prediction[0]

# --------------------------------
# CSE LIVE PRICE (API)
# --------------------------------
def get_live_price(symbol):
    try:
        base = symbol.replace(".CM", "")
        url = f"https://www.cse.lk/api/market-data/{base}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("lastTradedPrice", "N/A")
        return "N/A"
    except:
        return "N/A"

# --------------------------------
# SINGLE STOCK VIEW
# --------------------------------
symbol = st.text_input("Enter CSE Symbol (Example: JKH.CM)")

if symbol:
    data = get_data(symbol)

    if data is None or data.empty:
        st.error("No Data Found ❌")
    else:
        data['MA20'] = data['Close'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)

        st.subheader("📈 Price + MA")
        st.line_chart(data[['Close','MA20']])

        st.subheader("📊 RSI")
        st.line_chart(data['RSI'])

        latest_rsi = round(data['RSI'].iloc[-1],2)
        st.write("Latest RSI:", latest_rsi)

        if latest_rsi < 30:
            st.success("OVERSOLD (Buy Signal)")
        elif latest_rsi > 70:
            st.warning("OVERBOUGHT (Sell Signal)")
        else:
            st.info("Neutral Zone")

        # AI Prediction
        predicted_price = predict_price(data)
        st.subheader("🤖 AI 5-Day Future Prediction")
        st.write("Predicted Price:", round(predicted_price,2))

        # Live Price
        live_price = get_live_price(symbol)
        st.subheader("📡 CSE Live Price")
        st.write("Live Price:", live_price)

# --------------------------------
# MULTI STOCK SCANNER
# --------------------------------
st.subheader("🔥 Multi Stock Scanner")

scan_results = []

for stock in STOCK_LIST:
    data = get_data(stock)
    if data is not None and not data.empty:
        data['RSI'] = calculate_rsi(data)
        latest_rsi = data['RSI'].iloc[-1]

        if latest_rsi < 35:
            scan_results.append((stock, "Oversold"))

        elif latest_rsi > 65:
            scan_results.append((stock, "Overbought"))

if scan_results:
    for stock, signal in scan_results:
        st.write(stock, "→", signal)
else:
    st.write("No Strong Signals Found")
