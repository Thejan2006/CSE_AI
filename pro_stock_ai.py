import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# =================================================================
# CSE AI PRO ULTIMATE - 10-YEAR EXPERIENCED DEV EDITION
# =================================================================

st.set_page_config(page_title="CSE AI PRO ULTIMATE", page_icon="📈", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    /* Main background and text */
    .css-18e3th9 { background-color: #0e1117; color: #ffffff; }
    
    /* Metric styling */
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    
    /* Buttons */
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #00d4ff; color: black; font-weight: bold; }
    
    /* Headers */
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Inter', sans-serif; }
    
    /* Tabs & sidebar */
    .css-1v3fvcr { background-color: #0e1117; } /* sidebar background */
    </style>
""", unsafe_allow_html=True)

# --------------------------------
# CONFIGURATION
# --------------------------------
DEFAULT_TICKERS = {
    "John Keells (JKH)": "JKH.N0000",
    "Commercial Bank (COMB)": "COMB.N0000",
    "Sampath Bank (SAMP)": "SAMP.N0000",
    "Hayleys (HAYL)": "HAYL.N0000",
    "LOLC Holdings (LOLC)": "LOLC.N0000",
    "Aitken Spence (SPEN)": "SPEN.N0000",
    "Dialog Axiata (DIAL)": "DIAL.N0000",
    "Ceylon Tobacco (CTC)": "CTC.N0000"
}

# --------------------------------
# CORE ENGINE CLASSES
# --------------------------------

class CSEDataEngine:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_historical_data(ticker, period="2y"):
        try:
            data = yf.download(ticker, period=period, interval="1d", progress=False)
            if data.empty:
                return None
            return data
        except Exception as e:
            st.error(f"Fetch Error: {e}")
            return None

    @staticmethod
    def add_indicators(df):
        df = df.copy()
        # Moving Averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # Bollinger Bands
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df.dropna()

class CSAIPredictor:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_features(self):
        df = self.data.copy()
        df['Target'] = df['Close'].shift(-1)
        df['Vol'] = df['Close'].pct_change()
        features = ['Close', 'RSI', 'MACD', 'MA20', 'Vol']
        X = df[features].iloc[:-5].values
        y = df['Target'].iloc[:-5].values
        return X, y, features

    def train_and_predict(self, days_ahead=5):
        X, y, features = self.prepare_features()
        if len(X) < 20:
            return None, None
        self.model.fit(X, y)
        last_known = self.data[features].iloc[-1].values.reshape(1, -1)
        predictions = []
        current_input = last_known
        for _ in range(days_ahead):
            pred = self.model.predict(current_input)[0]
            predictions.append(pred)
            # Update next input with simple values
            current_input = np.array([[pred, 50, 0, pred, 0]])
        return predictions, self.model.score(X, y)

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("🇱🇰 CSE CONTROL CENTER")
selected_name = st.sidebar.selectbox("Select Stock", list(DEFAULT_TICKERS.keys()))
custom_ticker = st.sidebar.text_input("OR Enter Custom Ticker (e.g. JKH.N0000)")
active_ticker = custom_ticker if custom_ticker else DEFAULT_TICKERS[selected_name]

time_period = st.sidebar.select_slider(
    "Select Time Horizon",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    value="1y"
)

# --------------------------------
# MAIN DASHBOARD
# --------------------------------
st.title(f"📈 {selected_name} - AI Analysis")

engine = CSEDataEngine()
raw_data = engine.fetch_historical_data(active_ticker, period=time_period)

if raw_data is not None and not raw_data.empty:
    data = engine.add_indicators(raw_data)

    # Top Metrics
    last_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2])
    price_change = last_price - prev_price
    pct_change = (price_change / prev_price) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LATEST PRICE LKR", f"{last_price:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    m2.metric("RSI (14D)", f"{data['RSI'].iloc[-1]:.1f}")
    m3.metric("52W HIGH", f"{data['Close'].max():,.2f}")
    m4.metric("52W LOW", f"{data['Close'].min():,.2f}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price Action", "🤖 AI Prediction", "🔍 Tech Analysis"])

    with tab1:
        st.subheader("Price & Moving Averages")
        st.line_chart(data[['Close', 'MA20', 'MA50']])
        st.subheader("Volume Trend")
        st.bar_chart(data['Volume'])

    with tab2:
        st.subheader("AI Smart Forecast (Next 5 Trading Days)")
        predictor = CSAIPredictor(data)
        preds, score = predictor.train_and_predict()
        if preds:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.write("### Predicted Prices")
                future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 6)]
                forecast_df = pd.DataFrame({"Date": future_dates, "Forecast Price": preds})
                st.table(forecast_df.style.format({"Forecast Price": "{:.2f}"}))

                confidence = "High" if score > 0.8 else "Medium" if score > 0.6 else "Low"
                st.info(f"Model Training Accuracy: {score*100:.1f}%\nConfidence: **{confidence}**")
            with c2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data.index[-20:], data['Close'].iloc[-20:], label="Historical", color="#00d4ff")
                pred_indexes = [data.index[-1] + timedelta(days=i) for i in range(1, 6)]
                ax.plot(pred_indexes, preds, '--o', label="AI Forecast", color="#ffaa00")
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                ax.tick_params(colors='white')
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Insufficient data for AI training. Requires at least 20 trading days.")

    with tab3:
        st.subheader("Deep-Dive Oscillators")
        colA, colB = st.columns(2)
        with colA:
            st.write("**RSI (Relative Strength Index)**")
            st.line_chart(data['RSI'])
            if data['RSI'].iloc[-1] > 70: st.error("⚠️ OVERBOUGHT - Potential Reversal")
            elif data['RSI'].iloc[-1] < 30: st.success("✅ OVERSOLD - Potential Buy Opportunity")
            else: st.info("NEUTRAL ZONE")
        with colB:
            st.write("**MACD (Trend Momentum)**")
            st.line_chart(data[['MACD', 'Signal']])
            macd_signal = "BULLISH" if data['MACD'].iloc[-1] > data['Signal'].iloc[-1] else "BEARISH"
            if macd_signal == "BULLISH": st.success("📈 Momentum: BULLISH")
            else: st.error("📉 Momentum: BEARISH")

    # Summary & Signal
    st.divider()
    st.subheader("🎯 CSE PRO FINAL VERDICT")

    rsi_latest = data['RSI'].iloc[-1]
    macd_latest = data['MACD'].iloc[-1]
    signal_latest = data['Signal'].iloc[-1]

    sell_points = 0
    buy_points = 0
    if rsi_latest > 65: sell_points += 1
    if rsi_latest < 35: buy_points += 1
    if macd_latest > signal_latest: buy_points += 1
    else: sell_points += 1
    if last_price > data['MA20'].iloc[-1]: buy_points += 1
    else: sell_points += 1

    if buy_points >= 2:
        st.success(f"### STRONG BUY SIGNAL 🚀\nTechnical indicators suggest an upward trend for {active_ticker}.")
    elif sell_points >= 2:
        st.error(f"### STRONG SELL SIGNAL ⬇️\nCaution: Market indicators show bearish momentum for {active_ticker}.")
    else:
        st.warning(f"### HOLD / NEUTRAL ⚖️\nMarket is consolidating. No clear trend for {active_ticker}.")
else:
    st.error(f"❌ Could not retrieve data for {active_ticker}. Please check the symbol or internet connection.")
    st.info("Note: For Sri Lankan stocks, use the format SYMBOL.N0000 (e.g. JKH.N0000)")

# FOOTER
st.sidebar.divider()
st.sidebar.write("Developed by CSE AI PRO ULTIMATE")
st.sidebar.write("Role: 10-Year Python Dev Expert")
st.sidebar.caption("Data provided by Yahoo Finance API")

# CSV Export
if st.sidebar.button("💾 Export Analysis to CSV"):
    if 'data' in locals():
        data.to_csv(f"{active_ticker}_analysis.csv")
        st.sidebar.success(f"Saved to {active_ticker}_analysis.csv")