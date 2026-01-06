import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask, render_template, request
from utils.youtube import extract_video_id, get_comments
from utils.preprocess import preprocess_comment
import nltk
from utils.youtube import extract_video_id, get_comments
import joblib


# Load ML model and TF-IDF vectorizer
rf_model = joblib.load("model/model.pkl")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
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

    # TEMP: Just show sentiment percentages for now
    # return f"Sentiment percentages:<br>" \
    #        f"Positive: {sentiment_percentages['positive']}%<br>" \
    #        f"Neutral: {sentiment_percentages['neutral']}%<br>" \
    #        f"Negative: {sentiment_percentages['negative']}%"

    # Pass 10 sample comments to results page
    sample_comments = cleaned_comments[:100]

    return render_template("results.html", 
                           sentiment=sentiment_percentages,
                           comments=sample_comments)


    # # Pass to template
    # return render_template("results.html", comments=cleaned_comments[:100],
    #                        sentiment=sentiment_percentages)


if __name__ == "__main__":
    app.run(debug=True)
