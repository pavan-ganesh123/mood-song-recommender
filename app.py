import os
import pandas as pd
import joblib
import nltk
import traceback
from flask import Flask, render_template, request, session, jsonify, send_from_directory

# Download tokenizer if missing
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "64aac0bd88081458e38e9791196b2fb8"  # Change for production!

# --- Load model & vectorizer ---
try:
    model = joblib.load("best_mood_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"[ERROR] Model files not found: {e}")
    exit(1)

# --- Load songs dataset ---
songs_csv = "songs.csv"
if not os.path.exists(songs_csv):
    print(f"[ERROR] '{songs_csv}' not found!")
    exit(1)

try:
    songs_df = pd.read_csv(songs_csv, dtype=str)
except Exception as e:
    print(f"[ERROR] Failed loading CSV: {e}")
    exit(1)

# Directory for static songs
SONG_DIR = os.path.join(app.static_folder, "songs")
os.makedirs(SONG_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        print("✅ [DEBUG] Raw input:", data)

        # Expecting { "message": "..." }
        text = data.get("message", "").strip()
        print("✅ [DEBUG] Cleaned text:", text)

        if not text:
            return jsonify({"mood": "unknown", "songs": []})

        # Tokenize & vectorize
        tokens = word_tokenize(text.lower())
        tfidf = vectorizer.transform([" ".join(tokens)])
        print("✅ [DEBUG] TF-IDF shape:", tfidf.shape)

        # Predict
        raw_pred = model.predict(tfidf)[0]
        mood_map = {
            0: "sadness", 1: "joy", 2: "love",
            3: "anger", 4: "fear", 5: "surprise"
        }
        mood = mood_map.get(raw_pred, "unknown")
        print("✅ [DEBUG] Predicted mood:", mood)

        # Filter songs
        filtered = songs_df[songs_df["moods"].str.contains(mood, case=False, na=False)]
        print("✅ [DEBUG] Songs matched:", len(filtered))

        results = []
        for _, row in filtered.iterrows():
            fname = os.path.basename(row["file_path"].strip())
            full_path = os.path.join(SONG_DIR, fname)
            if not os.path.exists(full_path):
                print(f"⚠️ [WARN] Missing file: {full_path}")
                continue
            results.append({
                "song_name": row.get("song_name", fname),
                "file_path": f"/static/songs/{fname}"
            })

        # (Optional) store in session
        session["mood"] = mood
        session["songs"] = results
        session.modified = True

        return jsonify({"mood": mood, "songs": results})

    except Exception as e:
        print("❌ [ERROR] Exception in /api/predict:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/songs")
def songs_page():
    # If you want to use a Jinja template for songs list
    mood = request.args.get("mood", session.get("mood", "unknown"))
    songs = session.get("songs", [])
    return render_template("songs.html", mood=mood, songs=songs)


@app.route("/static/songs/<path:filename>")
def serve_song(filename):
    full_path = os.path.join(SONG_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": f"'{filename}' not found"}), 404
    return send_from_directory(SONG_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=True for detailed logs; disable in production
    app.run(host="0.0.0.0", port=port, debug=True)
