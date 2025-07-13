import praw
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import time

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="sentiment-tracker"
)

def get_reddit_posts(query=None, subreddit="stocks", days=7, limit=500):
    """
    Collect Reddit posts from a subreddit with improved filtering and error handling.
    
    Args:
        query: Search query (None for all posts)
        subreddit: Subreddit name
        days: Number of days to look back
        limit: Maximum number of posts to collect
    
    Returns:
        DataFrame with Reddit posts
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    posts = []
    
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        print(f"Accessing r/{subreddit}...")
        
        # If no query specified, get recent posts using different sorting methods
        if query is None:
            print("Collecting recent posts...")
            
            # Try multiple sorting methods to get more posts
            sorting_methods = [
                ('new', subreddit_obj.new(limit=limit//3)),
                ('hot', subreddit_obj.hot(limit=limit//3)),
                ('top', subreddit_obj.top(time_filter='week', limit=limit//3))
            ]
            
            seen_ids = set()
            
            for sort_name, submissions in sorting_methods:
                print(f"Fetching {sort_name} posts...")
                
                for submission in submissions:
                    # Skip if already seen
                    if submission.id in seen_ids:
                        continue
                    
                    seen_ids.add(submission.id)
                    
                    try:
                        created_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                        
                        # Filter by time range
                        if created_time < start_time:
                            continue
                        
                        # Skip deleted/removed posts
                        if submission.selftext == '[removed]' or submission.selftext == '[deleted]':
                            continue
                        
                        posts.append({
                            "id": submission.id,
                            "date": created_time.date(),
                            "created_utc": created_time,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "subreddit": subreddit,
                            "url": submission.url,
                            "permalink": submission.permalink,
                            "query": query
                        })
                        
                        # Add a small delay to avoid rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"Error processing submission {submission.id}: {e}")
                        continue
                
                print(f"Collected {len([p for p in posts if p not in posts[:posts.index(p)] if p in posts])} posts from {sort_name}")
        
        else:
            # Search with query
            print(f"Searching for: {query}")
            
            for submission in subreddit_obj.search(query, sort="new", time_filter="all", limit=limit):
                try:
                    created_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                    
                    if created_time < start_time:
                        continue
                    
                    if submission.selftext == '[removed]' or submission.selftext == '[deleted]':
                        continue
                    
                    posts.append({
                        "id": submission.id,
                        "date": created_time.date(),
                        "created_utc": created_time,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "subreddit": subreddit,
                        "url": submission.url,
                        "permalink": submission.permalink,
                        "query": query
                    })
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing submission {submission.id}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error accessing subreddit r/{subreddit}: {e}")
        return pd.DataFrame()
    
    print(f"Total posts collected: {len(posts)}")
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(posts)
    if not df.empty:
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df = df.sort_values('created_utc', ascending=False)
        df = df.drop_duplicates(subset=['id'])  # Remove any duplicates
        
        print(f"Final dataset: {len(df)} unique posts")
        print(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    
    return df

def get_reddit_comments(post_ids, limit_per_post=50):
    """
    Get comments for specific Reddit posts.
    
    Args:
        post_ids: List of Reddit post IDs
        limit_per_post: Maximum comments per post
    
    Returns:
        DataFrame with comments
    """
    comments = []
    
    for post_id in post_ids:
        try:
            submission = reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            
            for comment in submission.comments.list()[:limit_per_post]:
                if comment.body not in ['[removed]', '[deleted]']:
                    comments.append({
                        'post_id': post_id,
                        'comment_id': comment.id,
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                        'parent_id': comment.parent_id
                    })
                    
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching comments for post {post_id}: {e}")
            continue
    
    return pd.DataFrame(comments)

# Alternative function for when you want to collect ALL posts (not time-limited)
def get_all_reddit_posts(subreddit="elonmusk", limit=1000):
    """
    Collect all available posts from a subreddit without time restrictions.
    
    Args:
        subreddit: Subreddit name
        limit: Maximum number of posts to collect
    
    Returns:
        DataFrame with Reddit posts
    """
    posts = []
    
    try:
        subreddit_obj = reddit.subreddit(subreddit)
        print(f"Accessing r/{subreddit} for all posts...")
        
        # Get posts from multiple sources
        sources = [
            ('new', subreddit_obj.new(limit=limit//2)),
            ('hot', subreddit_obj.hot(limit=100)),
            ('top_week', subreddit_obj.top(time_filter='week', limit=100)),
            ('top_month', subreddit_obj.top(time_filter='month', limit=100)),
            ('top_year', subreddit_obj.top(time_filter='year', limit=limit//4))
        ]
        
        seen_ids = set()
        
        for source_name, submissions in sources:
            print(f"Fetching {source_name} posts...")
            count = 0
            
            for submission in submissions:
                if submission.id in seen_ids:
                    continue
                
                seen_ids.add(submission.id)
                
                try:
                    created_time = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                    
                    if submission.selftext == '[removed]' or submission.selftext == '[deleted]':
                        continue
                    
                    posts.append({
                        "id": submission.id,
                        "date": created_time.date(),
                        "created_utc": created_time,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "subreddit": subreddit,
                        "url": submission.url,
                        "permalink": submission.permalink,
                        "source": source_name
                    })
                    
                    count += 1
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error processing submission {submission.id}: {e}")
                    continue
            
            print(f"Added {count} posts from {source_name}")
    
    except Exception as e:
        print(f"Error accessing subreddit r/{subreddit}: {e}")
        return pd.DataFrame()
    
    print(f"Total posts collected: {len(posts)}")
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(posts)
    if not df.empty:
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df = df.sort_values('created_utc', ascending=False)
        df = df.drop_duplicates(subset=['id'])
        
        print(f"Final dataset: {len(df)} unique posts")
        print(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    
    return df