import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# Define directory structure
STOCKS_DIR = "data/stocks"
MUTUAL_FUNDS_DIR = "data/mutual_funds"

# Load saved models from specific folders
def load_model(ticker, model_type, is_mutual_fund):
    """Load a saved XGBoost or Prophet model for stocks or mutual funds from their specific folder."""
    folder = MUTUAL_FUNDS_DIR if is_mutual_fund else STOCKS_DIR
    file_path = os.path.join(folder, ticker, f"{ticker}_{model_type}_model.pkl")

    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        st.error(f"⚠️ Error: {ticker} {model_type} model file not found in {folder}.")
        return None

# Load historical data from specific folders
@st.cache_data
def load_data(ticker, is_mutual_fund=False):

    """Loads historical stock or mutual fund data from its specific folder."""
    folder = MUTUAL_FUNDS_DIR if is_mutual_fund else STOCKS_DIR
    file_path = os.path.join(folder, ticker, f"{ticker}_data.csv")

    try:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)  # ✅ Fixed datetime warning
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: {ticker} data file not found in {folder}. Please check if the file exists.")
        return None

# Streamlit UI
st.title("📈 Predictive Investment Insights Engine")
st.subheader("Stock & Mutual Fund Prediction Dashboard")

# User chooses between Stocks & Mutual Funds
investment_type = st.radio("Select Investment Type:", ["Stocks", "Mutual Funds"])

# **User input for stock or mutual fund selection**
if investment_type == "Stocks":
    options = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    stock_ticker = st.selectbox("Select a stock ticker:", options)
    data = load_data(stock_ticker, is_mutual_fund=False)
    is_mutual_fund = False
else:
    options = ["MFEGX", "MEIAX", "MFSIX", "MFBFX", "MWOIX"]
    stock_ticker = st.selectbox("Select an MFS mutual fund ticker:", options)
    data = load_data(stock_ticker, is_mutual_fund=True)
    is_mutual_fund = True

# Display selected investment information
st.write(f"### 📊 Analyzing: {stock_ticker}")

# Handle missing data
if data is None:
    st.error(f"❌ No data available for {stock_ticker}. Check CSV file.")
else:
    st.write("✅ **Data Loaded Successfully!**")
    st.write(data.tail(10))

# Show historical data if checkbox selected
if data is not None and st.checkbox("Show Full Historical Data"):
    st.write(data)

# **Predict next closing price using XGBoost**
# **Predict next closing price using XGBoost**
if data is not None and st.button("📉 Predict Next Closing Price (XGBoost)", key=f"predict_{stock_ticker}"):
    xgb_model = load_model(stock_ticker, "xgboost", is_mutual_fund)
    if xgb_model:
        try:
            latest_data = data.iloc[-1:][['Open', 'High', 'Low', 'Volume']]
            
            # 🔹 Ensure SMA_50, SMA_200, and RSI are included for stocks
            if not is_mutual_fund:
                missing_cols = [col for col in ['SMA_50', 'SMA_200', 'RSI'] if col not in data.columns]
                if missing_cols:
                    st.error(f"❌ Error: Missing features {missing_cols} in dataset for {stock_ticker}.")
                else:
                    latest_data = latest_data[['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'RSI']]

            # 🔹 Ensure no NaN values
            latest_data.fillna(0, inplace=True)

            prediction = xgb_model.predict(latest_data)[0]
            st.write(f"### 🎯 **Predicted Next Closing Price for {stock_ticker}:** **${prediction:.2f}**")
            st.success("✅ Prediction successful!")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# **Forecasting with Prophet**
if data is not None and st.button("🔮 Forecast with Prophet", key=f"forecast_{stock_ticker}"):
    prophet_model = load_model(stock_ticker, "prophet", is_mutual_fund)
    if prophet_model:
        try:
            future = prophet_model.make_future_dataframe(periods=365)
            forecast = prophet_model.predict(future)
            
            # Plot Forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual'))
            st.plotly_chart(fig)
            st.success("✅ Forecasting completed!")
        except Exception as e:
            st.error(f"Forecasting Error: {e}")
