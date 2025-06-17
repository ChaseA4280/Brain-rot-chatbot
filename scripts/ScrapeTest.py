# Test script - save this as test_scraping.py
import praw
import json

# Test your credentials
reddit = praw.Reddit(
    client_id="k4RKEdOhxfljGi0psvkg2w",
    client_secret="5fbHWbaT_h1USXTHH3ZqR1Kykv86MA", 
    user_agent="BrainRotBot/1.0 by Previous_Impress5216"
)

# Test with just one subreddit and a few posts
print("Testing Reddit connection...")
subreddit = reddit.subreddit('memes')

posts = []
for submission in subreddit.hot(limit=5):  # Just 5 posts to test
    print(f"Found: {submission.title[:50]}...")
    posts.append(submission.title)

print(f"Successfully scraped {len(posts)} posts!")
print("Your credentials work!")