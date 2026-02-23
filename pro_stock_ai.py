import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# Import our professional engine
from stock_engine import QualityValueEngine

# =================================================================
# PREMIER STOCK AI v2.0 - ENTERPRISE GRADE
# =================================================================

# 1. Page Configuration
st.set_page_config(
    page_title="QuantumTrade AI | Professional Stock Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0b0e14;
    }
    
    /* Header Animation */
    @keyframes glow {
        from { text-shadow: 0 0 10px #00f2ff55; }
        to { text-shadow: 0 0 20px #00f2ffaa; }
    }
    
    .premium-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2ff, #007bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00f2ff !important;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border: 1px solid #00f2ff;
        background: rgba(0, 242, 255, 0.02);
        transform: translateY(-5px);
    }
    
    /* Sidebar */
    .css-163n19o {
        background-color: #11151c !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #888;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 255, 0.1);
        color: #00f2ff !important;
        border-bottom: 2px solid #00f2ff;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Construction
st.sidebar.markdown("<h2 style='color:#00f2ff;'>💎 QuantumTrade AI</h2>", unsafe_allow_html=True)
st.sidebar.divider()

market_type = st.sidebar.radio("Select Market Focus", ["Global (USA)", "Sri Lanka (CSE)"])

if market_type == "Global (USA)":
    TICKER_MAP = {
        "Apple Inc. (AAPL)": "AAPL",
        "Tesla Motors (TSLA)": "TSLA",
        "NVIDIA Corp (NVDA)": "NVDA",
        "Microsoft (MSFT)": "MSFT",
        "Google (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Meta Platforms (META)": "META",
        "AMD (AMD)": "AMD"
    }
else:
    TICKER_MAP = {
        "John Keells (JKH)": "JKH.N0000",
        "Commercial Bank (COMB)": "COMB.N0000",
        "Sampath Bank (SAMP)": "SAMP.N0000",
        "Hayleys (HAYL)": "HAYL.N0000",
        "Aitken Spence (SPEN)": "SPEN.N0000"
    }

selected_name = st.sidebar.selectbox("Select Asset", list(TICKER_MAP.keys()))
custom_input = st.sidebar.text_input("🔍 Custom Symbol", placeholder="e.g. BTC-USD, MSTR, etc.")
active_ticker = custom_input.upper() if custom_input else TICKER_MAP[selected_name]

time_period = st.sidebar.select_slider(
    "Analysis Horizon",
    options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
    value="2y"
)

st.sidebar.divider()
st.sidebar.info("🤖 **Engine Status**: Active\n📡 **Data Feed**: Yahoo Finance\n🧠 **Model**: XGBoost Ensemble")

# 4. Main App Controller
def main():
    st.markdown(f"<div class='premium-header'>{selected_name if not custom_input else active_ticker} Analysis</div>", unsafe_allow_html=True)
    
    engine = QualityValueEngine()
    
    with st.spinner("Initializing Neural Ingestion..."):
        raw_df = engine.fetch_data(active_ticker, period=time_period)
    
    if raw_df.empty:
        st.error(f"Failed to fetch data for `{active_ticker}`. Possible reasons: Invalid ticker, connectivity issues, or market delisting.")
        return

    # Process Data
    df = engine.apply_technical_indicators(raw_df)
    
    # 5. Dashboard Header Metrics
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${last_close:,.2f}" if market_type=="Global (USA)" else f"Rs. {last_close:,.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    m2.metric("Volatility (ATR)", f"{df['ATR_14'].iloc[-1]:.2f}")
    m3.metric("RSI (14)", f"{df['RSI_14'].iloc[-1]:.1f}")
    m4.metric("Market Sentiment", "BULLISH" if last_close > df['EMA_50'].iloc[-1] else "BEARISH")

    # 6. Tabbed Visualization
    tab_price, tab_ai, tab_risk, tab_stats = st.tabs(["📊 Market Action", "🤖 AI Forecast", "🎲 Risk Analysis", "📚 Historical Data"])
    
    with tab_price:
        # Professional Candlestick Chart with Volume & EMA
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price Action & Indicators', 'Volume'), 
                           row_width=[0.2, 0.7])

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Candlestick"
        ), row=1, col=1)

        # EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#00f2ff', width=1.5), name="EMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#ffaa00', width=1.5), name="EMA 50"), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), name="BB Lower"), row=1, col=1)

        # Volume
        colors = ['red' if df['Open'].iloc[i] > df['Close'].iloc[i] else 'green' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        st.subheader("🚀 High-Confidence XGBoost Forecasting")
        
        c_pred, c_metrics = st.columns([2, 1])
        
        forecasts, conf_score = engine.train_and_forecast(df)
        
        if forecasts:
            with c_pred:
                # Forecast Visualization
                future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 6)]
                
                fig_ai = go.Figure()
                # Historical (Last 30 days)
                hist_zoom = df.tail(30)
                fig_ai.add_trace(go.Scatter(x=hist_zoom.index, y=hist_zoom['Close'], name="Recent History", line=dict(color="#888", width=2)))
                # Forecast
                fig_ai.add_trace(go.Scatter(x=future_dates, y=forecasts, name="AI Forecast", line=dict(color="#00f2ff", width=4, dash='dash'), mode='lines+markers'))
                
                fig_ai.update_layout(template="plotly_dark", height=450, title="Next 5-Day Forecast Projection")
                st.plotly_chart(fig_ai, use_container_width=True)
            
            with c_metrics:
                st.write("#### Prediction Confidence")
                st.progress(conf_score if conf_score > 0 else 0)
                st.write(f"Confidence Level: **{conf_score*100:.1f}%**")
                
                st.write("#### Numerical Forecast")
                forecast_df = pd.DataFrame({
                    "Date": [d.strftime('%Y-%m-%d') for d in future_dates],
                    "Target Price": [f"{v:,.2f}" for v in forecasts]
                })
                st.table(forecast_df)
                
                st.warning("Note: AI models thrive on patterns but market 'Black Swan' events can bypass any algorithm. Use for guidance only.")
        else:
            st.error("Insufficient data for training. The model requires at least 100 trading days of history.")

    with tab_risk:
        st.subheader("🎲 Monte Carlo Risk Simulation")
        st.write("Simulating 100 possible market paths for the next 30 days based on historical volatility.")
        
        sims = engine.monte_carlo_simulation(df)
        
        fig_mc = go.Figure()
        for col in sims.columns:
            fig_mc.add_trace(go.Scatter(y=sims[col], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
        
        fig_mc.update_layout(template="plotly_dark", height=500, xaxis_title="Days Ahead", yaxis_title="Price Path")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Risk Stats
        final_prices = sims.iloc[-1]
        var_95 = np.percentile(final_prices, 5)
        st.error(f"#### Value at Risk (VaR - 95%): ${var_95:,.2f}")
        st.info("95% of simulated outcomes stay above this price. This represents your potential 'downside' risk.")

    with tab_stats:
        st.subheader("📊 Core Feature Set")
        st.dataframe(df.tail(100).style.background_gradient(cmap='Blues'), use_container_width=True)
        
        csv = df.to_csv().encode('utf-8')
        st.download_button("💾 Download Full Analytical Dataset", data=csv, file_name=f"{active_ticker}_advanced_analysis.csv", mime='text/csv')

    # 7. Final Verdict System
    st.divider()
    v_col1, v_col2 = st.columns([3, 1])
    
    with v_col1:
        st.subheader("🎯 Quantitative Strategy Signal")
        
        # Signal Logic
        signals = {
            "RSI": "BULLISH" if df['RSI_14'].iloc[-1] < 40 else "BEARISH" if df['RSI_14'].iloc[-1] > 60 else "NEUTRAL",
            "EMA": "BULLISH" if df['Close'].iloc[-1] > df['EMA_50'].iloc[-1] else "BEARISH",
            "MACD": "BULLISH" if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] else "BEARISH",
            "Bollinger": "OVERSOLD" if df['Close'].iloc[-1] < df['BBL_20_2.0'].iloc[-1] else "OVERBOUGHT" if df['Close'].iloc[-1] > df['BBU_20_2.0'].iloc[-1] else "NEUTRAL"
        }
        
        bull_count = list(signals.values()).count("BULLISH") + list(signals.values()).count("OVERSOLD")
        bear_count = list(signals.values()).count("BEARISH") + list(signals.values()).count("OVERBOUGHT")
        
        if bull_count >= 3:
            st.success(f"### STRONG BUY SIGNAL 🚀\nMultiple technical layers (RSI, EMA, MACD) confirm a powerful bullish confluence for {active_ticker}.")
        elif bear_count >= 3:
            st.error(f"### STRONG SELL SIGNAL ⬇️\nCaution: Market structure has shifted. Significant overhead resistance and momentum deceleration detected.")
        else:
            st.warning(f"### HOLD / CONSOLIDATION ⚖️\nIndicators are conflicting. The asset is likely in a sideways range or accumulation phase.")

    with v_col2:
        st.write("#### Signal Breakdown")
        for k, v in signals.items():
            color = "#00ff88" if v in ["BULLISH", "OVERSOLD"] else "#ff4b4b" if v in ["BEARISH", "OVERBOUGHT"] else "#888"
            st.markdown(f"**{k}**: <span style='color:{color}'>{v}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()