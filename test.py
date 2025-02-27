import pandas as pd

ticker = "AAPL"  # Change this to test other stocks
file_path = f"data/stocks/{ticker}/{ticker}_data.csv"

df = pd.read_csv(file_path)

# Print available columns
print(f"Columns in {ticker} dataset: {df.columns}")

# Ensure indicators exist
missing_cols = [col for col in ["SMA_50", "SMA_200", "RSI"] if col not in df.columns]
if missing_cols:
    print(f"⚠️ Missing columns in {ticker} dataset: {missing_cols}")
else:
    print(f"✅ {ticker} dataset contains all required features.")
