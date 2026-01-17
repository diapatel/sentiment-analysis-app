from flask import Flask, render_template, request
from utils.youtube import extract_video_id, get_comments
from utils.preprocess import preprocess_comment
import joblib
import os

app = Flask(__name__)

# Lazy-loaded ML models
rf_model = None
tfidf_vectorizer = None

def load_models():
    """Load models only when needed"""
    global rf_model, tfidf_vectorizer
    if rf_model is None or tfidf_vectorizer is None:
        rf_model = joblib.load("model/model.pkl")
        tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    load_models()  # Load models safely here

    youtube_url = request.form.get("youtube_url")

    if not youtube_url:
        return "Please enter a YouTube URL."

    # Extract video ID
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL."

    # Fetch comments
    try:
        comments = get_comments(video_id)
    except Exception as e:
        return f"Error fetching comments: {e}"

    if not comments:
        return "No comments found for this video."

    # Preprocess comments
    cleaned_comments = [preprocess_comment(c) for c in comments if c]

    if not cleaned_comments:
        return "No valid comments to analyze after preprocessing."

    # Transform using TF-IDF vectorizer
    X = tfidf_vectorizer.transform(cleaned_comments)

    # Predict sentiment (0=negative, 1=neutral, 2=positive)
    predictions = rf_model.predict(X)

    # Map numeric predictions to labels
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    mapped_predictions = [label_map[p] for p in predictions]

    # Count sentiment categories
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for p in mapped_predictions:
        sentiment_counts[p] += 1

    # Convert counts to percentages
    total = len(mapped_predictions)
    sentiment_percentages = {k: round(v / total * 100, 2) for k, v in sentiment_counts.items()}

    # Pass sample comments to results page
    sample_comments = cleaned_comments[:100]

    return render_template("results.html", 
                           sentiment=sentiment_percentages,
                           comments=sample_comments)

# ✅ Render-ready app runner
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets $PORT automatically
    app.run(host="127.0.0.1", port=port)
