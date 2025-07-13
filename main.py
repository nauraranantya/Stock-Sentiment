from src.reddit_scraper import get_reddit_posts
from src.sentiment_analysis import analyze_sentiment
from src.stock_data import get_stock_data

import pandas as pd
import matplotlib.pyplot as plt

# Scrape Reddit + analyze sentiment
df = get_reddit_posts(query="Elon", subreddit="stocks", days=60)

if df.empty:
    print("No Reddit posts found!")
    exit()

df_sentiment = analyze_sentiment(df)

# Aggregate by date
df_sentiment['date'] = pd.to_datetime(df_sentiment['created_utc']).dt.date
daily_sentiment = df_sentiment.groupby('date')['compound'].mean().reset_index()
daily_sentiment.rename(columns={'compound': 'avg_sentiment'}, inplace=True)

# Get stock data
stock_df = get_stock_data("TSLA", days=60)

if stock_df.empty:
    print("No stock data found!")
    exit()

# Merge & plot
merged = pd.merge(daily_sentiment, stock_df, on='date')

if merged.empty:
    print("No overlapping dates between sentiment and stock data!")
    exit()

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('Avg Sentiment', color='tab:blue')
ax1.plot(merged['date'], merged['avg_sentiment'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('TSLA Close Price', color='tab:red')
ax2.plot(merged['date'], merged['stock_close'], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title("Reddit Sentiment vs TSLA Stock Price")
plt.tight_layout()
plt.show()

print(f"Successfully plotted data for {len(merged)} days")