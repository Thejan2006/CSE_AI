# Import libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------------
# 1) Get stock data using API
# ------------------------------
symbol = "AAPL"  # ඔබට වෙනස් කරන්න පුළුවන් (උදා: "TSLA", "MSFT", "GOOGL")
start_date = "2015-01-01"
end_date   = "2025-01-01"

# Download historical stock data
df = yf.download(symbol, start=start_date, end=end_date)
print(f"Data fetched: {df.shape} rows")
print(df.head())

# ------------------------------
# 2) Preprocess data (scale)
# ------------------------------
close_prices = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# ------------------------------
# 3) Create training sequences
# ------------------------------
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        y.append(dataset[i+look_back])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)
X = X.reshape(X.shape[0], X.shape[1], 1)

# ------------------------------
# 4) Build LSTM model
# ------------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back,1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------------
# 5) Train model
# ------------------------------
print("Training model...please wait")
model.fit(X, y, batch_size=32, epochs=20)

# ------------------------------
# 6) Predict next 30 days
# ------------------------------
test_data = scaled_data[-look_back:]
test_input = test_data.reshape(1, look_back, 1)
predictions = []
current_input = test_input

for _ in range(30):
    pred = model.predict(current_input, verbose=0)[0]
    predictions.append(pred)
    current_input = np.append(current_input[:, 1:, :], [[pred]], axis=1)

pred_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# ------------------------------
# 7) Plot historical + predicted prices
# ------------------------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"], label="Historical Close")
future_dates = pd.date_range(df.index[-1], periods=31, freq="B")[1:]
plt.plot(future_dates, pred_prices, label="Predicted Prices")
plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 8) Save predictions to CSV
# ------------------------------
output = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": pred_prices.flatten()
})
output.to_csv(f"{symbol}_predictions.csv", index=False)
print(f"Saved predictions to {symbol}_predictions.csv")
