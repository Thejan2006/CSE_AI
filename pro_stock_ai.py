# ============================================================
# Sri Lanka / Global Stock Price Predictor using LSTM
# Compatible with TensorFlow 2.16+ / Keras 3.x
# ============================================================

# Import libraries
import os
import io
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN info messages
# Fix Windows console encoding issues (charmap codec)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Use standalone Keras 3 (works with TF 2.16+)
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ---------------------------------------------------------------
# 1) Configuration - change the symbol to any stock you want
#
#    NOTE on Sri Lanka CSE stocks:
#    Yahoo Finance has very limited coverage of CSE (Colombo Stock
#    Exchange) tickers. Most CSE tickers (e.g. "JKH.N0000") are
#    NOT available via yfinance. To predict CSE stocks you would
#    need a local CSV file with historical prices.
#
#    For now the script uses a globally-traded stock (AAPL).
#    You can change SYMBOL to any Yahoo Finance ticker:
#      "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", etc.
# ---------------------------------------------------------------
SYMBOL     = "AAPL"   # <-- change this to your preferred stock
START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"
LOOK_BACK  = 60            # number of past days used for each prediction
EPOCHS     = 20
BATCH_SIZE = 32
PREDICT_DAYS = 30          # how many future days to predict

# ---------------------------------------------------------------
# 2) Download historical stock data
# ---------------------------------------------------------------
print(f"\n[INFO] Downloading data for: {SYMBOL}")
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True)

if df.empty:
    raise ValueError(
        f"No data found for symbol '{SYMBOL}'. "
        "Check the ticker symbol and date range."
    )

# yfinance ≥0.2.x may return a MultiIndex column – flatten it
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"[OK] Data fetched: {df.shape[0]} rows  |  Columns: {list(df.columns)}")
print(df.tail())

# ---------------------------------------------------------------
# 3) Preprocess – scale Close prices to [0, 1]
# ---------------------------------------------------------------
close_prices = df["Close"].values.reshape(-1, 1)   # shape: (N, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ---------------------------------------------------------------
# 4) Create training sequences
# ---------------------------------------------------------------
def create_dataset(dataset, look_back=60):
    """
    Converts a time-series array into (X, y) pairs.
    X[i] = dataset[i : i+look_back]   (shape: look_back × 1)
    y[i] = dataset[i + look_back]
    """
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i : i + look_back])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, LOOK_BACK)
# X shape: (samples, look_back, 1)  – already correct for LSTM
print(f"\n[INFO] Training samples: {X.shape[0]}  |  X shape: {X.shape}")

# ---------------------------------------------------------------
# 5) Build LSTM model
# ---------------------------------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# ---------------------------------------------------------------
# 6) Train the model
# ---------------------------------------------------------------
print("\n[INFO] Training model - please wait...")
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# ---------------------------------------------------------------
# 7) Predict the next PREDICT_DAYS trading days
# ---------------------------------------------------------------
# Seed the prediction with the last LOOK_BACK days of scaled data
current_input = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)  # (1, 60, 1)
predictions = []

for _ in range(PREDICT_DAYS):
    pred = model.predict(current_input, verbose=0)[0, 0]   # scalar
    predictions.append(pred)
    # Slide the window: drop oldest value, append new prediction
    new_val = np.array([[[pred]]])                          # (1, 1, 1)
    current_input = np.concatenate(
        [current_input[:, 1:, :], new_val], axis=1
    )                                                       # (1, 60, 1)

# Inverse-transform back to original price scale
pred_prices = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

# ---------------------------------------------------------------
# 8) Plot historical + predicted prices
# ---------------------------------------------------------------
future_dates = pd.date_range(df.index[-1], periods=PREDICT_DAYS + 1, freq="B")[1:]

plt.figure(figsize=(14, 6))
plt.plot(df.index, df["Close"], label="Historical Close", color="#1f77b4")
plt.plot(future_dates, pred_prices, label=f"Predicted (next {PREDICT_DAYS} days)",
         color="#ff7f0e", linestyle="--", marker="o", markersize=3)
plt.title(f"{SYMBOL} – Stock Price Prediction (LSTM)", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{SYMBOL.replace('.', '_')}_prediction.png", dpi=150)
plt.show()
print(f"[OK] Chart saved as {SYMBOL.replace('.', '_')}_prediction.png")

# ---------------------------------------------------------------
# 9) Save predictions to CSV
# ---------------------------------------------------------------
output_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": pred_prices.flatten()
})
csv_name = f"{SYMBOL.replace('.', '_')}_predictions.csv"
output_df.to_csv(csv_name, index=False)
print(f"\n[OK] Predictions saved to: {csv_name}")
print(output_df.to_string(index=False))
