from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # Add sentiment scores
    sentiments = df['text'].apply(analyzer.polarity_scores).tolist()
    sentiment_df = pd.DataFrame(sentiments)
    
    # Combine with original data
    df_sentiment = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    return df_sentiment