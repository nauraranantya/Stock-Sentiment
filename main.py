import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')
from src.reddit_scraper import get_reddit_posts, get_all_reddit_posts
from src.stock_data import get_stock_data
from src.finbert_sentiment import analyze_finbert_sentiment

# Create output directory for figures
os.makedirs("figures", exist_ok=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- Phase 1: Data Collection & Preprocessing ---
print("Phase 1: Collecting and Preprocessing Data...")

# Reddit data - Try multiple approaches
print("Attempting to collect Reddit posts...")

reddit_df = get_reddit_posts(query=None, subreddit="teslamotors", days=100, limit=1000)

if reddit_df.empty:
    print("ERROR: Could not collect any Reddit posts. Please check:")
    print("1. Your Reddit API credentials in .env file")
    print("2. Internet connection")
    print("3. Reddit API rate limits")
    print("4. Subreddit accessibility")
    exit(1)

print(f"Successfully collected {len(reddit_df)} Reddit posts.")

# Display sample of collected data
print("\nSample of collected posts:")
print(reddit_df[['date', 'title', 'score']].head())

# Check if we have the required columns
required_columns = ['title', 'selftext']
missing_columns = [col for col in required_columns if col not in reddit_df.columns]
if missing_columns:
    print(f"ERROR: Missing required columns: {missing_columns}")
    exit(1)

# Clean the data
print("Cleaning Reddit data...")
reddit_df['title'] = reddit_df['title'].fillna('')
reddit_df['selftext'] = reddit_df['selftext'].fillna('')

# Remove posts with empty title and selftext
reddit_df = reddit_df[~((reddit_df['title'] == '') & (reddit_df['selftext'] == ''))]

if reddit_df.empty:
    print("ERROR: No valid posts remaining after cleaning.")
    exit(1)

print(f"After cleaning: {len(reddit_df)} posts remain.")

# Analyze sentiment using FinBERT
print("Analyzing sentiment with FinBERT...")
try:
    finbert_df = analyze_finbert_sentiment(reddit_df)
    os.makedirs("data/reddit", exist_ok=True)
    finbert_df.to_csv("data/reddit/elon_finbert_sentiment.csv", index=False)
    print("Sentiment analysis completed and saved.")
except Exception as e:
    print(f"Error in sentiment analysis: {e}")
    exit(1)

# --- Phase 2: Enhanced Sentiment Aggregation ---
print("\nPhase 2: Aggregating Sentiment Data with Daily Averages...")

sentiment_df = finbert_df.copy()
sentiment_df['created_utc'] = pd.to_datetime(sentiment_df['created_utc'])
sentiment_df['date'] = sentiment_df['created_utc'].dt.date

# Map sentiment labels to numerical scores
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
sentiment_df['sentiment_score'] = sentiment_df['label'].map(sentiment_mapping)

# Check if we have enough data for daily aggregation
print(f"Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
print(f"Number of unique dates: {sentiment_df['date'].nunique()}")

if sentiment_df['date'].nunique() < 2:
    print("WARNING: Only data from one day. Analysis may be limited.")

# Group by date and calculate statistics
grouped = sentiment_df.groupby('date')

# Calculate each statistic separately and convert to proper format
sentiment_avg = grouped['sentiment_score'].mean().reset_index()
sentiment_std = grouped['sentiment_score'].std().reset_index()
post_count = grouped['sentiment_score'].count().reset_index()
finbert_confidence_mean = grouped['score'].mean().reset_index() # Renamed variable

daily_stats = sentiment_avg.copy()
daily_stats = daily_stats.rename(columns={'sentiment_score': 'sentiment_avg'})
daily_stats = daily_stats.merge(sentiment_std.rename(columns={'sentiment_score': 'sentiment_std'}), on='date', how='left')
daily_stats = daily_stats.merge(post_count.rename(columns={'sentiment_score': 'post_count'}), on='date', how='left')
daily_stats = daily_stats.merge(finbert_confidence_mean.rename(columns={'score': 'finbert_confidence'}), on='date', how='left')

# Calculate sentiment percentages
sentiment_counts = grouped['label'].value_counts().unstack(fill_value=0)
sentiment_counts = sentiment_counts.reindex(columns=['positive', 'negative', 'neutral'], fill_value=0)
sentiment_counts = sentiment_counts.reset_index()

# Merge with daily_stats
daily_stats = daily_stats.merge(sentiment_counts, on='date', how='left')

# Rename columns for clarity
daily_stats = daily_stats.rename(columns={
    'positive': 'positive_count',
    'negative': 'negative_count', 
    'neutral': 'neutral_count'
})

# Calculate percentages
daily_stats['total_posts'] = daily_stats['post_count']
daily_stats['%pos'] = daily_stats['positive_count'] / daily_stats['total_posts']
daily_stats['%neg'] = daily_stats['negative_count'] / daily_stats['total_posts']
daily_stats['%neu'] = daily_stats['neutral_count'] / daily_stats['total_posts']

# Calculate sentiment momentum (change from previous day)
daily_stats = daily_stats.sort_values('date')
daily_stats['sentiment_momentum'] = daily_stats['sentiment_avg'].diff()

# Fill NaN values in sentiment_std with 0 (happens when there's only one post per day)
daily_stats['sentiment_std'] = daily_stats['sentiment_std'].fillna(0)

print("Enhanced sentiment aggregation completed.")

# --- Phase 3: Stock Data & Merging ---
print("\nPhase 3: Fetching Stock Data and Merging...")

try:
    stock_df = get_stock_data("TSLA", days=100)
    os.makedirs("data/stocks", exist_ok=True)
    stock_df.to_csv("data/stocks/tsla.csv", index=False)
    
    # Calculate additional stock metrics
    stock_df['daily_return'] = (stock_df['Close'] - stock_df['Open']) / stock_df['Open']
    stock_df['volatility'] = abs(stock_df['High'] - stock_df['Low']) / stock_df['Open']
    stock_df['price_change'] = stock_df['Close'].pct_change()
    stock_df['volume_ma'] = stock_df['Volume'].rolling(window=3).mean()
    stock_df['volume_spike'] = stock_df['Volume'] / stock_df['volume_ma']
    
    print("Stock data fetched and enhanced metrics calculated.")
    
except Exception as e:
    print(f"Error fetching stock data: {e}")
    exit(1)

# Merge data
print("Merging sentiment and stock data...")
merged = pd.merge(daily_stats, stock_df, on='date', how='inner')

if merged.empty:
    print("WARNING: No overlapping dates between sentiment and stock data.")
    print("Sentiment data date range:", daily_stats['date'].min(), "to", daily_stats['date'].max())
    print("Stock data date range:", stock_df['date'].min(), "to", stock_df['date'].max())
    
    # Try outer join to see what data we have
    merged = pd.merge(daily_stats, stock_df, on='date', how='outer')
    print(f"Outer merge resulted in {len(merged)} rows.")
    
    if merged.empty:
        print("ERROR: Could not merge data.")
        exit(1)

os.makedirs("data/merged", exist_ok=True)
merged.to_csv("data/merged/tesla_sentiment_stock_enhanced.csv", index=False)
print("Enhanced sentiment and stock data merged and saved.")

# --- Phase 4: Advanced Analysis & Visualization ---
print("\nPhase 4: Performing Advanced Analysis and Visualization...")

if len(merged) < 2:
    print("WARNING: Insufficient data for comprehensive analysis. Creating basic visualizations...")

merged_df = merged.copy()
merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df.set_index('date', inplace=True)

# Remove rows with all NaN values
merged_df = merged_df.dropna(how='all')

if merged_df.empty:
    print("ERROR: No valid data after merging.")
    exit(1)

# 1. Basic visualizations that work with minimal data
print("\nGenerating visualizations...")

# Create a flexible plot based on available data
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Available sentiment and stock data
ax1 = axes[0]
ax1_twin = ax1.twinx()

# Only plot if we have the data
if 'Close' in merged_df.columns and not merged_df['Close'].isna().all():
    line1 = ax1.plot(merged_df.index, merged_df['Close'], color='#1f77b4', linewidth=2, label='TSLA Close Price')
    ax1.set_ylabel('Stock Price (USD)', fontsize=12, color='#1f77b4')

if 'sentiment_avg' in merged_df.columns and not merged_df['sentiment_avg'].isna().all():
    line2 = ax1_twin.plot(merged_df.index, merged_df['sentiment_avg'], color='#ff7f0e', linewidth=2, label='Daily Sentiment Average')
    ax1_twin.set_ylabel('Sentiment Average (-1 to 1)', fontsize=12, color='#ff7f0e')

ax1.set_title('TSLA Stock Price vs Daily Sentiment Average', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Plot 2: Sentiment distribution
ax2 = axes[1]
if 'sentiment_avg' in merged_df.columns and not merged_df['sentiment_avg'].isna().all():
    ax2.hist(merged_df['sentiment_avg'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Sentiment Average')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Daily Sentiment Averages', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/basic_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# Calculate basic statistics
print("\n=== BASIC STATISTICS ===")
if 'sentiment_avg' in merged_df.columns:
    print(f"Average sentiment: {merged_df['sentiment_avg'].mean():.3f}")
    print(f"Sentiment std dev: {merged_df['sentiment_avg'].std():.3f}")
    print(f"Number of data points: {len(merged_df['sentiment_avg'].dropna())}")

if 'daily_return' in merged_df.columns and 'sentiment_avg' in merged_df.columns:
    correlation = merged_df['daily_return'].corr(merged_df['sentiment_avg'])
    print(f"Correlation (sentiment vs daily return): {correlation:.3f}")

# Only do advanced analysis if we have sufficient data
if len(merged_df) >= 5:
    print("\nGenerating advanced analysis...")
    
    # 2. CORRELATION ANALYSIS
    correlation_vars = ['sentiment_avg', 'daily_return', 'Close', 'Volume']
    available_vars = [var for var in correlation_vars if var in merged_df.columns and not merged_df[var].isna().all()]
    
    if len(available_vars) >= 2:
        correlation_df = merged_df[available_vars].dropna()
        
        if len(correlation_df) >= 2:
            pearson_corr = correlation_df.corr(method='pearson')
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, fmt='.3f', 
                       square=True, linewidths=.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig("figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Correlation analysis completed.")

print("\nAnalysis Complete!")
print("Generated files:")
print("- basic_analysis.png")
if len(merged_df) >= 5:
    print("- correlation_heatmap.png")
print("- Data files in data/ directory")

# Save summary statistics
summary_stats = {
    'total_posts': len(reddit_df),
    'date_range': f"{reddit_df['date'].min()} to {reddit_df['date'].max()}",
    'sentiment_avg': merged_df['sentiment_avg'].mean() if 'sentiment_avg' in merged_df.columns else None,
    'sentiment_std': merged_df['sentiment_avg'].std() if 'sentiment_avg' in merged_df.columns else None,
    'data_points': len(merged_df)
}

with open("data/summary_stats.txt", "w") as f:
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")

print(f"\nSummary saved to data/summary_stats.txt")