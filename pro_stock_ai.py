#comment is work



import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

class CSEProAnalyzer:

    def __init__(self, symbol):
        self.symbol = symbol
        self.data = yf.download(symbol, period="3y", interval="1d")
        if self.data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

    def technical_analysis(self):
        df = self.data
        df['RSI'] = ta.momentum.rsi(df['Close'], 14)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], 50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], 200)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['BB_HIGH'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_LOW'] = ta.volatility.bollinger_lband(df['Close'])
        df.dropna(inplace=True)
        self.data = df
        return df

    def random_forest_prediction(self):
        df = self.data.copy()
        X = df[['RSI','SMA_50','SMA_200','MACD']]
        y = df['Close']

        model = RandomForestRegressor(n_estimators=200)
        model.fit(X[:-30], y[:-30])

        prediction = model.predict(X[-1:].values)
        return prediction[0]

    def lstm_prediction(self):
        df = self.data.copy()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[['Close']])

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i])
            y.append(scaled[i])


        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1],1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_60 = scaled[-60:]
        last_60 = np.reshape(last_60,(1,60,1))
        pred = model.predict(last_60)
        return scaler.inverse_transform(pred)[0][0]

    def risk_score(self):
        df = self.data
        volatility = df['Close'].pct_change().std()
        if volatility < 0.02:
            return "LOW RISK"
        elif volatility < 0.05:
            return "MEDIUM RISK"
        else:
            return "HIGH RISK"

    def full_report(self):
        self.technical_analysis()
        rf_pred = self.random_forest_prediction()
        lstm_pred = self.lstm_prediction()
        risk = self.risk_score()

        current_price = self.data['Close'].iloc[-1]
        avg_pred = (rf_pred + lstm_pred)/2

        print(f"\nStock: {self.symbol}")
        print(f"Current Price: {current_price}")
        print(f"AI Predicted Price: {avg_pred}")
        print(f"Risk Level: {risk}")

        if avg_pred > current_price:
            print("Signal: BUY Probability High")
        else:
            print("Signal: SELL / WAIT")

# Usage
analyzer = CSEProAnalyzer("BIL.N0000")
analyzer.full_report()
