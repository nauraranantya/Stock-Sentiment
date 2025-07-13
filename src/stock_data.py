import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(ticker, days=60):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # Reset index to make 'Date' a regular column
    df = df.reset_index()
    
    # If there are multiple levels in columns, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Rename columns to match your main.py expectations
    df = df.rename(columns={
        'Date': 'date',
        'Close': 'stock_close'
    })
    
    # Convert date to date object (not datetime) to match sentiment data
    df['date'] = df['date'].dt.date
    
    # Keep only the columns you need
    df = df[['date', 'stock_close']]
    
    return df