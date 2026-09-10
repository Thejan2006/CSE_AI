import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up professional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityValueEngine:
    """
    Professional-grade Financial Data & AI Engine.
    Custom implementation optimized for Python 3.14+ (Dependency-light).
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

    def fetch_data(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical data with error handling and validation."""
        try:
            logger.info(f"Fetching data for {ticker}...")
            df = yf.download(ticker, period=period, interval="1d", progress=False)
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Standardize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def apply_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual implementation of technical indicators to avoid dependency issues on Python 3.14."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = exp1 - exp2
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # Bollinger Bands
        df['BBM_20_2.0'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['BBM_20_2.0'] + (std * 2)
        df['BBL_20_2.0'] = df['BBM_20_2.0'] - (std * 2)
        
        # ATR (Volatility)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        df['ATR_14'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1).rolling(window=14).mean()
        
        return df.dropna()

    def prepare_ml_features(self, df: pd.DataFrame):
        """Prepare feature set for RandomForest forecasting."""
        df_feat = df.copy()
        
        # Target: Price in 5 days
        df_feat['Target'] = df_feat['Close'].shift(-5)
        
        features = [
            'Close', 'EMA_20', 'EMA_50', 'EMA_200', 
            'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9',
            'BBU_20_2.0', 'BBL_20_2.0', 'ATR_14'
        ]
        
        data = df_feat.dropna()
        if len(data) < 50:
            return None, None, None, None, []
            
        X = data[features]
        y = data['Target']
        
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        return X_train, X_test, y_train, y_test, features

    def train_and_forecast(self, df: pd.DataFrame, days_to_forecast: int = 5):
        """Train model and generate future forecast."""
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_ml_features(df)
        
        if X_train is None:
            return None, 0
        
        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        
        last_params = df[feature_cols].tail(1)
        forecasts = []
        current_data = last_params.copy()
        
        # Simple projection
        for _ in range(days_to_forecast):
            pred = self.model.predict(current_data)[0]
            forecasts.append(pred)
            current_data['Close'] = pred  # Recursive update for visual trend
            
        return forecasts, score

    def monte_carlo_simulation(self, df: pd.DataFrame, days: int = 30, iterations: int = 50):
        """Professional Risk Analysis using Monte Carlo simulations."""
        returns = df['Close'].pct_change().dropna()
        last_price = df['Close'].iloc[-1]
        
        simulation_df = pd.DataFrame()
        for i in range(iterations):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * (1 + np.random.choice(returns)))
            simulation_df[f'sim_{i}'] = prices
            
        return simulation_df
