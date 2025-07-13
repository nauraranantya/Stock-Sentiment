import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone

def get_stock_data(ticker="TSLA", days=30):
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    df = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.reset_index()
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'date'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date']).dt.date
    desired_columns = ['date', 'Open', 'Close', 'High', 'Low', 'Volume']
    
    existing_desired_columns = [col for col in desired_columns if col in df.columns]
    
    if len(existing_desired_columns) < len(desired_columns):
        missing_cols = set(desired_columns) - set(existing_desired_columns)
        print(f"\n--- Warning: Missing desired stock data columns: {missing_cols} ---")

    df = df[existing_desired_columns]

    return df