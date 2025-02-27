import os
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Ensure directory structure exists
def ensure_directory(directory):
    """Creates the necessary directories if they do not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def fetch_data(ticker, start_date, end_date, is_mutual_fund=False):
    """Fetch historical stock or mutual fund data from Yahoo Finance and save it in the correct folder."""
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    if data.empty:
        print(f"⚠️ Warning: No historical data found for {ticker}. Skipping CSV save.")
        return None

    # Ensure Date column exists for merging
    data.reset_index(inplace=True)

    # Fetch the latest available data
    latest = stock.history(period="1d")
    if not latest.empty:
        latest_row = {
            "Date": latest.index[-1].strftime("%Y-%m-%d"),
            "Open": latest["Open"].values[-1],
            "High": latest["High"].values[-1],
            "Low": latest["Low"].values[-1],
            "Close": latest["Close"].values[-1],
            "Volume": latest["Volume"].values[-1],
            "Dividends": 0,
            "Stock Splits": 0
        }
        latest_df = pd.DataFrame([latest_row])
        data = pd.concat([data, latest_df], ignore_index=True)

    # **Ensure SMA & RSI are computed for stocks**
    if not is_mutual_fund:  # FIX: Properly indent inside the condition
        data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['SMA_200'] = data['Close'].rolling(window=200, min_periods=1).mean()
        data['RSI'] = calculate_rsi(data['Close'])

    # Fill missing values
    data.fillna(method="ffill", inplace=True)
    data.fillna(method="bfill", inplace=True)

    # **Ensure there are no missing values before saving**
    if data.isnull().sum().sum() > 0:
        print(f"⚠️ Warning: {ticker} data contains missing values after processing!")

    # Define save directory
    data_type = "mutual_funds" if is_mutual_fund else "stocks"
    directory = f"data/{data_type}/{ticker}"
    ensure_directory(directory)

    # Save data to CSV
    filename = f"{directory}/{ticker}_data.csv"
    data.to_csv(filename, index=False)
    print(f"✅ {ticker} data saved successfully to {filename} with {len(data)} rows.")

    return data


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI) for a given stock price series."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_xgboost_model(data, ticker, is_mutual_fund=False):
    """Train an XGBoost model to predict stock closing price and save it in the correct folder."""
    
    if data is None or data.empty:
        print(f"⚠️ Skipping XGBoost training for {ticker} due to missing data.")
        return None

    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Volume']
    
    # Stocks include technical indicators, mutual funds do not
    if 'SMA_50' in data.columns:
        features.extend(['SMA_50', 'SMA_200', 'RSI'])

    # Check if all required columns exist
    missing_features = [col for col in features if col not in data.columns]
    if missing_features:
        print(f"⚠️ Skipping {ticker} training, missing features: {missing_features}")
        return None

    X = data[features]
    y = data['Target']

    if len(X) < 50:
        print(f"⚠️ Skipping training for {ticker}, not enough data points.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"  {ticker} XGBoost Model MAE:", mean_absolute_error(y_test, predictions))

    # Define save directory
    data_type = "mutual_funds" if is_mutual_fund else "stocks"
    directory = f"data/{data_type}/{ticker}"
    ensure_directory(directory)

    # Save the model
    joblib.dump(model, f"{directory}/{ticker}_xgboost_model.pkl")
    return model

def train_prophet_model(data, ticker, is_mutual_fund=False):
    """Train a Prophet model to forecast stock or mutual fund NAV prices and save it in the correct folder."""
    
    if data is None or data.empty:
        print(f"⚠️ Skipping Prophet training for {ticker} due to missing data.")
        return None, None

    prophet_df = data[['Date', 'Close']].copy()  # Ensure 'Date' column exists
    prophet_df.columns = ['ds', 'y']  # Rename for Prophet
    
    # Convert 'ds' to datetime format before Prophet training
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')

    # Remove timezone information
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Define save directory
    data_type = "mutual_funds" if is_mutual_fund else "stocks"
    directory = f"data/{data_type}/{ticker}"
    ensure_directory(directory)

    # Save the model
    joblib.dump(model, f"{directory}/{ticker}_prophet_model.pkl")
    print(f" {ticker} Prophet model trained and saved!")

    return model, forecast

if __name__ == "__main__":
    # Stock tickers
    tickers_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]  
    
    # MFS Mutual Funds (Example tickers, may need verification)
    tickers_funds = ["MFEGX", "MEIAX", "MFSIX", "MFBFX", "MWOIX"]  

    start = "2000-01-01"
    end = "2024-01-01"

    # Process Stocks
    for ticker in tickers_stocks:
        print(f"Processing Stock: {ticker}...")
        stock_data = fetch_data(ticker, start, end, is_mutual_fund=False)

        if stock_data is None:
            continue

        xgb_model = train_xgboost_model(stock_data, ticker, is_mutual_fund=False)
        prophet_model, prophet_forecast = train_prophet_model(stock_data, ticker, is_mutual_fund=False)
        print(f" {ticker} models trained and saved successfully!")

    # Process MFS Mutual Funds
    for ticker in tickers_funds:
        print(f"Processing Mutual Fund: {ticker}...")
        fund_data = fetch_data(ticker, start, end, is_mutual_fund=True)

        if fund_data is None:
            continue

        xgb_model = train_xgboost_model(fund_data, ticker, is_mutual_fund=True)
        prophet_model, prophet_forecast = train_prophet_model(fund_data, ticker, is_mutual_fund=True)
        print(f"  {ticker} models trained and saved successfully!")
