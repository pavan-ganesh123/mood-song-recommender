import os
import pandas as pd
import joblib
import pygame
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize

# Load the trained model and vectorizer
model = joblib.load("best_mood_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load songs dataset
songs_df = pd.read_csv("songs.csv")

# Initialize pygame mixer
pygame.mixer.init()

def play_song(file_path):
    """Play a song using pygame."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return False

    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    print(f"Now Playing: {os.path.basename(file_path)}")
    
    while pygame.mixer.music.get_busy():
        continue  # Wait until the song is finished playing
    
    return True  # Proceed to next song

def predict_mood_and_get_songs(message):
    """Predict mood from message and return matching song paths."""
    processed_text = [" ".join(word_tokenize(message.lower()))]
    text_tfidf = vectorizer.transform(processed_text)
    
    predicted_mood = model.predict(text_tfidf)[0]
    mood_name = {0: "sadness", 1: "joy", 2: "love", 3: "anger"}.get(predicted_mood, "unknown")
    
    filtered_songs = songs_df[songs_df["moods"].str.contains(mood_name, case=False, na=False)]
    
    if filtered_songs.empty:
        return {"mood": mood_name, "songs": []}
    
    return {"mood": mood_name, "songs": filtered_songs["file_path"].tolist()}

# Flask Backend
app = Flask(__name__)

@app.route("/predict_mood", methods=["GET"])
def predict_mood():
    """API Endpoint: Predict mood and return song list."""
    message = request.args.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    result = predict_mood_and_get_songs(message)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
