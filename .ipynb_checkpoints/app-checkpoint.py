from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify
import os
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK package
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "64aac0bd88081458e38e9791196b2fb8"  # Secure this in production

# Load Model and Vectorizer with error handling
try:
    model = joblib.load("best_mood_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError as e:
    print(f"Error: Model files not found! {e}")
    exit(1)

# Load Songs Dataset with error handling
songs_csv_path = "songs.csv"
if not os.path.exists(songs_csv_path):
    print(f"Error: '{songs_csv_path}' not found!")
    exit(1)

try:
    songs_df = pd.read_csv(songs_csv_path)
except Exception as e:
    print(f"Error loading songs CSV: {e}")
    exit(1)

# Define the directory where songs are stored (inside static folder)
SONG_DIRECTORY = os.path.join(app.static_folder, "songs")

# Ensure the song directory exists
os.makedirs(SONG_DIRECTORY, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        return redirect(url_for("songs", mood="unknown"))

    try:
        # Tokenize and preprocess the input text
        processed_text = [" ".join(word_tokenize(text.lower()))]
        text_tfidf = vectorizer.transform(processed_text)

        # Predict the mood
        predicted_mood = model.predict(text_tfidf)[0]
        print(f"Predicted Mood (Raw): {predicted_mood}")  # Debugging

        mood_mapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
        mood_name = mood_mapping.get(predicted_mood, "unknown")
        print(f"Predicted Mood (Mapped): {mood_name}")  # Debugging

        # Filter songs matching the predicted mood
        filtered_songs = songs_df[songs_df["moods"].str.contains(mood_name, case=False, na=False)]
        
        song_list = []
        if not filtered_songs.empty:
            for _, song in filtered_songs.iterrows():
                file_name = os.path.basename(song["file_path"].strip())
                song_path = f"{request.host_url}static/songs/{file_name}"

                song_list.append({
                    "song_name": song["song_name"],
                    "file_path": song_path
                })

        # Store the mood and songs in the session
        session["mood"] = mood_name
        session["songs"] = song_list  
        session.modified = True  # Ensure Flask saves session data

        return redirect(url_for("songs", mood=mood_name))

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/songs")
def songs():
    # Get the mood from the query parameters or session
    mood = request.args.get("mood", session.get("mood", "unknown"))
    songs = session.get("songs", [])

    print(f"Accessing /songs with mood: {mood}")
    print(f"Session Songs (Before Rendering): {songs}")

    return render_template("songs.html", mood=mood, songs=songs)

@app.route("/songs/<path:filename>")
def serve_song(filename):
    """Serve song files dynamically from the 'static/songs' directory."""
    try:
        # Ensure the file exists
        if not os.path.exists(os.path.join(SONG_DIRECTORY, filename)):
            print(f"Error: File '{filename}' not found in '{SONG_DIRECTORY}'")
            return jsonify({"error": f"File '{filename}' not found"}), 404

        return send_from_directory(SONG_DIRECTORY, filename)
    except Exception as e:
        print(f"Error serving song: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)