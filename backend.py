import os
import pandas as pd
import joblib
import nltk
import traceback
from flask import Flask, request, jsonify, url_for

# Ensure the NLTK punkt tokenizer is available
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

# Initialize the Flask app
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

# Load the trained model and TF‑IDF vectorizer
try:
    model = joblib.load("best_mood_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"[ERROR] Model/vectorizer not found: {e}")
    exit(1)

# Load the songs dataset
if not os.path.exists("songs.csv"):
    print("[ERROR] songs.csv missing")
    exit(1)

try:
    songs_df = pd.read_csv("songs.csv", dtype=str)
except Exception as e:
    print(f"[ERROR] Failed to load songs.csv: {e}")
    exit(1)

# Ensure the directory for static song files exists
SONG_DIR = os.path.join(app.static_folder, "songs")
os.makedirs(SONG_DIR, exist_ok=True)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    POST /api/predict
    Expects JSON: { "message": "I feel amazing today!" }
    Returns JSON:
      {
        "mood": "joy",
        "songs": [
          { "song_name": "Bliss", "url": "/static/songs/bliss.mp3" },
          ...
        ]
      }
    """
    try:
        # Parse incoming JSON
        data = request.get_json(force=True)
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in JSON body"}), 400

        text = data["message"].strip()
        if not text:
            return jsonify({"mood": "unknown", "songs": []})

        # Tokenize and vectorize
        tokens = word_tokenize(text.lower())
        tfidf_vector = vectorizer.transform([" ".join(tokens)])

        # Predict mood
        raw_pred = model.predict(tfidf_vector)[0]
        mood_map = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        mood = mood_map.get(raw_pred, "unknown")

        # Filter songs by mood
        matches = songs_df[
            songs_df["moods"].str.contains(mood, case=False, na=False)
        ]

        results = []
        for _, row in matches.iterrows():
            fname = os.path.basename(row["file_path"].strip())
            full_path = os.path.join(SONG_DIR, fname)
            if not os.path.exists(full_path):
                print(f"[WARN] Missing file: {full_path}")
                continue
            results.append({
                "song_name": row.get("song_name", fname),
                "url": url_for('static', filename=f"songs/{fname}", _external=False)
            })

        return jsonify({"mood": mood, "songs": results})

    except Exception as e:
        print("[ERROR] Exception in /api/predict:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Bind to 0.0.0.0 on the port provided by the environment (Render, etc.)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
