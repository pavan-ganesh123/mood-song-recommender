from flask import Flask, render_template, request, send_from_directory, session, jsonify
import os
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK package (will skip if already present)
nltk.download('punkt', quiet=True)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "64aac0bd88081458e38e9791196b2fb8"  # Secure this in production!

# Load Model and Vectorizer
try:
    model = joblib.load("best_mood_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"[ERROR] Model files not found: {e}")
    exit(1)

# Load Songs Dataset
songs_csv_path = "songs.csv"
if not os.path.exists(songs_csv_path):
    print(f"[ERROR] '{songs_csv_path}' not found!")
    exit(1)

try:
    songs_df = pd.read_csv(songs_csv_path)
except Exception as e:
    print(f"[ERROR] Failed loading CSV: {e}")
    exit(1)

# Directory where song files live
SONG_DIRECTORY = os.path.join(app.static_folder, "songs")
os.makedirs(SONG_DIRECTORY, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1) Parse JSON body
        data = request.get_json(force=True)
        text = data.get("text", "").strip()

        # 2) If empty, return empty result
        if not text:
            return jsonify({"mood": "unknown", "songs": []})

        # 3) Preprocess & vectorize
        tokens = word_tokenize(text.lower())
        processed_text = [" ".join(tokens)]
        tfidf = vectorizer.transform(processed_text)

        # 4) Predict mood
        raw_pred = model.predict(tfidf)[0]
        mood_mapping = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise"
        }
        mood_name = mood_mapping.get(raw_pred, "unknown")

        # 5) Filter songs by mood
        filtered = songs_df[
            songs_df["moods"].str.contains(mood_name, case=False, na=False)
        ]

        song_list = []
        for _, row in filtered.iterrows():
            fname = os.path.basename(row["file_path"].strip())
            # Relative URL for the browser
            rel_url = f"/static/songs/{fname}"
            song_list.append({
                "song_name": row["song_name"],
                "file_path": rel_url
            })

        # (Optional) store in session if you still need it
        session["mood"] = mood_name
        session["songs"] = song_list
        session.modified = True

        # 6) Return JSON
        return jsonify({"mood": mood_name, "songs": song_list})

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/songs")
def songs():
    """ If you still want a /songs page rendering via Jinja """
    mood = request.args.get("mood", session.get("mood", "unknown"))
    songs = session.get("songs", [])
    return render_template("songs.html", mood=mood, songs=songs)

@app.route("/songs/<path:filename>")
def serve_song(filename):
    """ Serve files from static/songs """
    full_path = os.path.join(SONG_DIRECTORY, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"'{filename}' not found"}), 404
    return send_from_directory(SONG_DIRECTORY, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or fallback to 5000
    app.run(host="0.0.0.0", port=port, debug=True)