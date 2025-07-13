from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

def load_finbert_pipeline():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline


def analyze_finbert_sentiment(df):
    if 'title' not in df.columns:
        raise ValueError("Expected a 'title' column.")

    pipe = load_finbert_pipeline()

    texts = df['title'].fillna("").astype(str) + " " + df['selftext'].fillna("").astype(str)
    sentiments = []
    
    for t in tqdm(texts, desc="Analyzing Sentiment"):
        result = pipe(t[:512])[0]  # truncate to 512 tokens
        sentiments.append({
            "label": result['label'],  # POSITIVE / NEGATIVE / NEUTRAL
            "score": result['score']
        })

    sent_df = pd.DataFrame(sentiments)
    result_df = pd.concat([df.reset_index(drop=True), sent_df], axis=1)
    return result_df