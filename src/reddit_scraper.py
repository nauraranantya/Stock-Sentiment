import praw
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="sentiment-tracker"
)

def get_reddit_posts(query="Elon", subreddit="stocks", days=7, limit=500):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    posts = []
    for submission in reddit.subreddit(subreddit).search(query, sort="new", time_filter="all", limit=limit):
        created_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created_time < start_time:
            continue
        posts.append({
            "id": submission.id,
            "date": created_time.date(),
            "created_utc": created_time,
            "title": submission.title,
            "selftext": submission.selftext,
            "subreddit": subreddit,
            "query": query
        })

    return pd.DataFrame(posts)