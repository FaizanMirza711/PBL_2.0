import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import numpy as np
import parselmouth
import pymysql


# Initialize the recognizer and sentiment analyzer
recognizer = sr.Recognizer()
analyzer = SentimentIntensityAnalyzer()

# Use the default microphone as the audio source
microphone = sr.Microphone()

# Adjust the microphone sensitivity dynamically
recognizer.dynamic_energy_threshold = True

print("Listening...")

# Exit phrase
exit_phrase = "terminate the session"

# Connect to MySQL database
conn = pymysql.connect(
    host="localhost",
    port=3306,
    user="root",
    password="nitant@123",
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

# Function to insert data into the database
def insert_data(timestamp, text, sentiment, pitch):
    sql = "INSERT INTO speech_sentiments (timestamp, text, sentiment, pitch) VALUES (%s, %s, %s, %s)"
    val = (timestamp, text, sentiment, pitch)
    cursor.execute(sql, val)  
    conn.commit()

# Function to calculate pitch using parselmouth


# Function to calculate pitch from audio samples using parselmouth
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


# Continuously listen for speech input
with microphone as source:
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source)
    
    while True:
        try:
            # Capture the audio from the microphone
            audio = recognizer.listen(source, timeout=None)
            
            # Convert speech to text
            text = recognizer.recognize_google(audio)
            
            # Get current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Print the recognized text with timestamp
            print(f"[{timestamp}] You said:", text)
            
            # Perform sentiment analysis using VADER
            vader_sentiment = analyzer.polarity_scores(text)
            sentiment = vader_sentiment['compound']  # Compound score gives a good overall sentiment value
            
            # Convert the audio data to numpy array for parselmouth
            audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
            
            # Calculate pitch using parselmouth
            pitch = calculate_pitch(audio_data)
            
            # Adjust sentiment based on pitch deviation
            baseline_pitch = 180  # You can adjust this or compute dynamically based on average
            pitch_deviation = pitch - baseline_pitch
            pitch_sensitivity = 0.005  # Adjust for higher sensitivity
            sentiment += pitch_sensitivity * pitch_deviation
            
            # Classify sentiment based on the adjusted score
            if sentiment > 0.6:
                print("Surprised")
            elif sentiment > 0.3:
                print("Happy")
            elif sentiment < -0.6:
                print("Angry or Disgusted")
            elif sentiment < -0.3:
                print("Sad")
            elif -0.2 < sentiment < 0.2:
                print("Neutral")
            else:
                print("Mildly Positive/Negative")
            
            # Store data in the database
            insert_data(timestamp, text, sentiment, pitch)
            
            # Check if the exit phrase is spoken
            if any(phrase in text.lower() for phrase in ["terminate the session", "end session", "stop listening"]):
                print("Exiting program...")
                break
  # Break out of the loop to end the program
        
        except sr.UnknownValueError:
            # Speech is unintelligible
            print("Sorry, I couldn't understand what you said.")
        
        except sr.RequestError as e:
            # Error occurred with the speech recognition service
            print(f"Could not request results from Google Speech Recognition service; {e}")


conn.close()
