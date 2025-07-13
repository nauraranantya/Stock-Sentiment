import praw
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

def get_reddit_posts(subreddit="wallstreetbets", query="Tesla", days=7, limit=500):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    posts = []
    for submission in reddit.subreddit(subreddit).search(query, sort="new", time_filter="all", limit=limit):
        created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
        if created < start_time:
            continue
        posts.append({
            "created_utc": created,
            "text": submission.title + " " + submission.selftext
        })

    return pd.DataFrame(posts)