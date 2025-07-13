from src.reddit_scraper import get_reddit_posts
from src.stock_data import get_stock_data
from src.finbert_sentiment import analyze_finbert_sentiment
import pandas as pd
import os

# Fetch stock data
stock_df = get_stock_data("TSLA", days=30)
os.makedirs("data/stocks", exist_ok=True)
stock_df.to_csv("data/stocks/tsla.csv", index=False)

# Reddit data
reddit_df = get_reddit_posts(query="Elon", subreddit="stocks", days=30, limit=500)

# Analyze sentiment using FinBERT
finbert_df = analyze_finbert_sentiment(reddit_df)
finbert_df.to_csv("data/reddit/elon_finbert_sentiment.csv", index=False)

# Load FinBERT output
sentiment_df = finbert_df
sentiment_df['created_utc'] = pd.to_datetime(sentiment_df['created_utc']).dt.date

# group by date and label
daily = sentiment_df.groupby(['created_utc', 'label']).size().unstack(fill_value=0)

for col in ['positive', 'negative', 'neutral']:
    if col not in daily.columns:
        daily[col] = 0

# daily percentages
daily['total'] = daily[['positive', 'negative', 'neutral']].sum(axis=1)
daily['%pos'] = daily['positive'] / daily['total']
daily['%neg'] = daily['negative'] / daily['total']
daily['%neu'] = daily['neutral'] / daily['total']

daily_sentiment = daily[['%pos', '%neg', '%neu']].reset_index().rename(columns={'created_utc': 'date'})



# Merge 
merged = pd.merge(daily_sentiment, stock_df, on='date', how='inner')
merged.to_csv("data/merged/tesla_sentiment_stock.csv", index=False)

print(merged.head())