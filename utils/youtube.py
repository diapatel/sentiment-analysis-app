import os
from googleapiclient.discovery import build
import re

# Get your API key from environment variable
API_KEY = os.environ.get("YOUTUBE_API_KEY")

def extract_video_id(url):
    """
    Extract YouTube video ID from different URL formats
    """
    patterns = [
        r"v=([^&]+)",       # for URLs like https://www.youtube.com/watch?v=VIDEO_ID
        r"youtu\.be/([^?]+)"  # for URLs like https://youtu.be/VIDEO_ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_comments(video_id, max_results=100):
    """
    Fetch top-level comments from a YouTube video
    """
    youtube = build("youtube", "v3", developerKey=API_KEY)
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    
    response = request.execute()
    
    comments = []
    
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments
