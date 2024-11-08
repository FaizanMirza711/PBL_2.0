import os
import pandas as pd
import numpy as np
import speech_recognition as sr
import parselmouth
import pymysql
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------- DATA LOADING FUNCTIONS --------------------

# Function to load data from a folder (positive/negative)

import os
import pandas as pd

base_dir = r'C:\\Users\\nitan\\Desktop\\College\\Psychsense\\PYCODES\\datasetimdb'

def load_data_from_folder(folder_path, label):
    data = []
    print("Looking for files in:", folder_path)  # Debugging statement
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            data.append((file.read(), label))
    return pd.DataFrame(data, columns=['review', 'sentiment'])

def load_imdb_dataset(base_dir):
    # Load training data
    train_pos = load_data_from_folder(os.path.join(base_dir, 'train', 'pos'), label=1)
    train_neg = load_data_from_folder(os.path.join(base_dir, 'train', 'neg'), label=0)
    train_data = pd.concat([train_pos, train_neg])

    # Load testing data
    test_pos = load_data_from_folder(os.path.join(base_dir, 'test', 'pos'), label=1)
    test_neg = load_data_from_folder(os.path.join(base_dir, 'test', 'neg'), label=0)
    test_data = pd.concat([test_pos, test_neg])

    # Shuffle the data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data

train_data, test_data = load_imdb_dataset(base_dir)

# -------------------- SENTIMENT MODEL TRAINING FUNCTION --------------------

def train_sentiment_model(train_data, test_data):
    # Preprocessing text (TF-IDF Vectorization)
    vectorizer = TfidfVectorizer(max_features=5000)

    # Fit on training data and transform both train and test sets
    X_train_tfidf = vectorizer.fit_transform(train_data['review'])
    X_test_tfidf = vectorizer.transform(test_data['review'])

    y_train = train_data['sentiment']
    y_test = test_data['sentiment']

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model and vectorizer for later use
    joblib.dump(model, "sentiment_model_imdb.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer_imdb.pkl")

    return model, vectorizer

# -------------------- PITCH ANALYSIS FUNCTION --------------------

# Function to calculate pitch using parselmouth
def calculate_pitch(audio_data, sample_rate=44100):
    # Create a Sound object from audio data
    sound = parselmouth.Sound(audio_data, sample_rate)
    
    # Extract pitch using parselmouth
    pitch = sound.to_pitch()

    # Get pitch values as a numpy array
    pitch_values = pitch.selected_array['frequency']
    
    # Filter out unvoiced values (0 Hz) for more accurate pitch calculation
    voiced_pitch_values = pitch_values[pitch_values > 0]

    # Calculate the mean pitch from voiced pitch values
    if len(voiced_pitch_values) > 0:
        mean_pitch = np.mean(voiced_pitch_values)
    else:
        mean_pitch = 0  # No voiced pitches detected, set mean to zero

    return mean_pitch

# -------------------- DATABASE FUNCTIONS --------------------

# Function to insert data into the MySQL database
def insert_data(timestamp, text, sentiment, pitch, cursor, conn):
    sql = "INSERT INTO speech_sentiments (timestamp, text, sentiment, pitch) VALUES (%s, %s, %s, %s)"
    val = (timestamp, text, sentiment, pitch)
    cursor.execute(sql, val)
    conn.commit()

# -------------------- REAL-TIME SPEECH-TO-TEXT FUNCTION --------------------

def real_time_speech_analysis():
    # Load the pre-trained sentiment analysis model and vectorizer
    model = joblib.load("sentiment_model_imdb.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_imdb.pkl")

    # Initialize the recognizer
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Connect to MySQL database
    conn = pymysql.connect(
        host="localhost",
        port=3306,
        user="root",
        password="your_password",
        database="speechtotext"
    )
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS speech_sentiments (
            timestamp TEXT, 
            text TEXT, 
            sentiment FLOAT,
            pitch FLOAT
        )
    ''')

    # Continuously listen for speech input
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            try:
                # Capture the audio from the microphone
                audio = recognizer.listen(source, timeout=None)
                
                # Convert speech to text
                text = recognizer.recognize_google(audio)
                
                # Get current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
                print(f"[{timestamp}] You said:", text)
                
                # Analyze sentiment using the trained model
                text_tfidf = vectorizer.transform([text])
                sentiment = model.predict(text_tfidf)[0]

                # Print sentiment classification
                if sentiment == 1:
                    print("Positive Sentiment")
                else:
                    print("Negative Sentiment")
                
                # Convert audio data to numpy array for parselmouth
                audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                pitch = calculate_pitch(audio_data)
                
                # Store data in the database
                insert_data(timestamp, text, sentiment, pitch, cursor, conn)
            
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand what you said.")
            
            except sr.RequestError as e:
                print(f"Error with Google Speech Recognition service: {e}")

    conn.close()

# -------------------- RUN THE CODE --------------------

# Step 1: Load the IMDb dataset (adjust the base directory path)
base_dir = 'aclImdb/'
train_data, test_data = load_imdb_dataset(base_dir)

# Step 2: Train the sentiment model
train_sentiment_model(train_data, test_data)

# Step 3: Run the real-time speech-to-text sentiment analysis
real_time_speech_analysis()
