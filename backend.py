import os
import pandas as pd
import joblib
import nltk
from flask import Flask, request, jsonify, url_for

# Ensure punkt is available
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Load model & vectorizer
try:
    model = joblib.load("best_mood_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"[ERROR] Model/vectorizer not found: {e}")
    exit(1)

# Load songs dataset
if not os.path.exists("songs.csv"):
    print("[ERROR] songs.csv missing")
    exit(1)

songs_df = pd.read_csv("songs.csv", dtype=str)

# Location of your static songs
SONG_DIR = os.path.join(app.static_folder, "songs")
os.makedirs(SONG_DIR, exist_ok=True)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    Expects JSON: { "message": "I feel amazing today!" }
    Returns JSON: 
      {
        "mood": "joy",
        "songs": [
          { "song_name":"Bliss", "url":"/static/songs/bliss.mp3" },
          ...
        ]
      }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in JSON body"}), 400

    text = data["message"].strip()
    if not text:
        return jsonify({"mood":"unknown","songs":[]})

    # Tokenize & vectorize
    tokens = word_tokenize(text.lower())
    tf = vectorizer.transform([" ".join(tokens)])
    raw = model.predict(tf)[0]

    # Map to mood
    mood_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    mood = mood_map.get(raw, "unknown")

    # Filter songs
    matches = songs_df[
        songs_df["moods"].str.contains(mood, case=False, na=False)
    ]

    results = []
    for _, row in matches.iterrows():
        fname = os.path.basename(row["file_path"].strip())
        static_path = f"songs/{fname}"
        # Verify exists
        full = os.path.join(SONG_DIR, fname)
        if not os.path.exists(full):
            print(f"[WARN] missing file: {full}")
            continue
        results.append({
            "song_name": row.get("song_name", fname),
            "url": url_for('static', filename=static_path, _external=False)
        })

    return jsonify({"mood": mood, "songs": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=True only for development
    app.run(host="0.0.0.0", port=port, debug=True)
